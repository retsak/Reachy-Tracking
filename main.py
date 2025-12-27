
import cv2
import threading
import time
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np

from robot_controller import RobotController
from detection_engine import DetectionEngine
from simple_tracker import SimpleTracker

try:
    cv2.setUseOptimized(True)
except Exception:
    pass

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals
robot = RobotController()
detector = DetectionEngine() # Now uses YOLOv8 (Output is list of dicts)
current_status = "System Initializing..."
SERVER_STATE = {
    "paused": True, 
    "wiggle_enabled": True
} # Default Paused, Wiggle ON
LATEST_CANDIDATES = []

# State
latest_frame = None
latest_camera_frame = None
latest_camera_lock = threading.Lock()

# DNN Result State
# Detections: list of {'box':(x,y,w,h), 'label':str, 'conf':float}
latest_dnn = {"detected": False, "detections": [], "ts": 0.0}
latest_dnn_lock = threading.Lock()
is_running = True
last_detection_time = 0

# Tuning
TUNING = {
    "detection_interval": 0.20, # Run DNN ~5Hz (YOLOv8n is fast)
    "command_interval": 1.2,    # 1.2s moves
    "stream_fps_cap": 60.0,
    "min_score_threshold": 280.0
}
JPEG_QUALITY = 70

def _create_tracker():
    """Create a lightweight OpenCV tracker (KCF/MOSSE) for the primary target."""
    candidates = [
        ("legacy", "TrackerKCF_create"), # KCF is stable for single target
        ("legacy", "TrackerMOSSE_create"),
        (None, "TrackerKCF_create"),
    ]
    for module_name, factory_name in candidates:
        try:
            mod = getattr(cv2, module_name) if module_name else cv2
            factory = getattr(mod, factory_name, None)
            if factory:
                return factory()
        except Exception:
            continue
    return None

def detection_loop():
    """Run YOLOv8 in background."""
    global latest_camera_frame
    while is_running:
        with latest_camera_lock:
            frame = None if latest_camera_frame is None else latest_camera_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue
        
        # Detector now returns list of dicts
        results = detector.detect(frame) 
        
        now = time.time()
        with latest_dnn_lock:
            latest_dnn["detected"] = (len(results) > 0)
            latest_dnn["detections"] = results
            latest_dnn["ts"] = now

        time.sleep(TUNING["detection_interval"])

def connection_monitor_loop():
    global current_status
    while is_running:
        if not robot.is_connected:
            current_status = "Connecting to Robot..."
            robot.connect(silent=True)
            if robot.is_connected:
                current_status = "Robot Connected."
        time.sleep(3.0)

def video_stream_loop():
    global latest_frame, last_detection_time, current_status, latest_camera_frame
    
    last_seen_time = time.time()
    head_recentered = False
    
    # Tracking Logic
    mot_tracker = SimpleTracker(max_disappeared=5, max_distance=100)
    current_target_id = None # ID of the object we are focusing on
    
    # Fast Visual Tracker (KCF) for the chosen target
    visual_tracker = None
    
    # Config
    FOV_X, FOV_Y = 60.0, 45.0
    GAIN_YAW, GAIN_PITCH = 0.7, 0.7
    
    last_dnn_ts = 0.0
    last_move_end_time = 0.0
    
    # Smoothing
    smooth_x, smooth_y = None, None
    ALPHA = 0.3 
    
    last_command_time = 0.0
    target_present = False
    RECENTER_TIMEOUT = 2.0
    
    while is_running:
        loop_start = time.perf_counter()
        current_time = time.time()
        
        # 1. Get Frame
        frame = robot.get_latest_frame()
        if frame is not None:
            with latest_camera_lock:
                latest_camera_frame = frame
        else:
            # Handle No Video
            time.sleep(0.01)
            continue

        frame_h, frame_w = frame.shape[:2]
        
        # 2. Check DNN Updates
        new_dnn_data = None
        with latest_dnn_lock:
            # ONLY accept new DNN data if it was captured AFTER the last move finished.
            # This ensures we don't correct based on frames taken *during* movement (blur/shift).
            if latest_dnn["ts"] > last_dnn_ts and latest_dnn["ts"] > last_move_end_time:
                new_dnn_data = latest_dnn["detections"]
                last_dnn_ts = latest_dnn["ts"]
        
        # 3. Update MOT Tracker (Scene Understanding)
        trackable_objects = mot_tracker.objects 
        
        if new_dnn_data is not None:
            trackable_objects = mot_tracker.update(new_dnn_data)
            
            # --- RANKING LOGIC ---
            # Pre-calc bounding boxes for persons for context validation
            person_boxes = [obj['data'][0] for obj in trackable_objects.values() if obj['data'][1] == 'person']

            best_id = None
            best_score = -1.0
            current_target_score = -1.0
            debug_candidates = []
            current_frame_candidates = []
            
            for obj_id, obj in trackable_objects.items():
                box, label, conf = obj['data']
                (x, y, w, h) = box
                cx, cy = obj['centroid']
                
                # Base Score Components
                score = 0.0
                prio_score = 0.0
                context_score = 0.0
                
                # A. Type Priority (BOOSTED BASES)
                # SWAPPED: Person > Face (User prefers Orange/Confident YOLO over Green/Random Haar)
                if label == 'person': 
                    score += 150.0 
                    prio_score = 150.0

                elif label == 'face': 
                    score += 200.0
                    prio_score = 120.0
                    # E. Context Validation (Realness Check)
                    is_validated = False
                    for (px, py, pw, ph) in person_boxes:
                        # Check: Centroid inside box AND in top 60% of height (Head area)
                        if (cx >= px and cx <= (px + pw) and 
                            cy >= py and cy <= (py + ph * 0.6)):
                            is_validated = True
                            break
                    
                    if is_validated:
                        score += 50.0 
                        context_score = 50.0
                    else:
                        score -= 60.0 
                        context_score = -60.0

                elif label == 'cat': 
                    score += 150.0 # Cat is still high priority
                    prio_score = 150.0
                
                # F. Confidence
                conf_bonus = (conf * 50.0)
                score += conf_bonus

                # B. Centrality
                dx = cx - (frame_w / 2)
                dy = cy - (frame_h / 2)
                dist = np.sqrt(dx*dx + dy*dy)
                max_dist = np.sqrt((frame_w/2)**2 + (frame_h/2)**2)
                cent_bonus = (1.0 - (dist / max_dist)) * 40.0
                score += cent_bonus
                
                # C. Size
                area = w * h
                frame_area = frame_w * frame_h
                size_bonus = min(1.0, area / (frame_area * 0.5)) * 40.0
                score += size_bonus
                
                # D. Stickiness (REMOVED - Using explicit threshold logic instead)
                # Capture Score of Current Target if present
                if current_target_id is not None and obj_id == current_target_id:
                    current_target_score = score
                
                # Structured Data for UI
                cand_data = {
                    "id": obj_id,
                    "label": label,
                    "score": int(score),
                    "priority": int(prio_score),
                    "context": int(context_score),
                    "confidence": int(conf_bonus),
                    "centrality": int(cent_bonus),
                    "size": int(size_bonus)
                }
                # debug_candidates.append(cand_data) <-- REMOVED caused TypeError
                current_frame_candidates.append(cand_data)

                debug_string = f"ID{obj_id}({label}) S:{score:.0f} [P:{prio_score} Ctx:{context_score} Cnf:{conf_bonus:.0f} Cnt:{cent_bonus:.0f} Sz:{size_bonus:.0f}]"
                debug_candidates.append(debug_string)

                if score > best_score:
                    best_score = score
                    best_id = obj_id
            
            # Update Global State
            global LATEST_CANDIDATES
            LATEST_CANDIDATES = current_frame_candidates
            
            # --- EXPLICIT SWITCHING LOGIC ---
            # Requirement: New Score > Current Score + 50 (unless current is lost)
            
            # 1. Check Threshold for Best Candidate
            if best_score < TUNING["min_score_threshold"]:
                 best_id = None
            
            # 2. Apply Hysteresis
            if current_target_id is not None and current_target_id in trackable_objects:
                # Find current target's score (Captured in loop above)
                if current_target_score > 0:
                     # If best_id is DIFFERENT from current, check the delta.
                     if best_id is not None and best_id != current_target_id:
                         if best_score <= (current_target_score + 50.0):
                             # Challenger is not strong enough. Hold current.
                             # print(f"Holding ID {current_target_id} (S:{current_target_score:.0f}). Challenger ID {best_id} (S:{best_score:.0f}) didn't exceed by 50.")
                             best_id = current_target_id 
                
             # --- END HYSTERESIS ---
            
            # Switch Target logic...
            if best_id is not None:
                if best_id != current_target_id:
                     print(f"*** SWITCH TARGET: {current_target_id} -> {best_id} ***\nCandidates: {', '.join(debug_candidates)}\n*************************")
                     current_target_id = best_id
                     tgt_box = trackable_objects[best_id]['data'][0]
                     visual_tracker = _create_tracker()
                     if visual_tracker:
                         visual_tracker.init(frame, tuple(tgt_box))
                     visual_tracker = _create_tracker()
                     if visual_tracker:
                         visual_tracker.init(frame, tuple(tgt_box))
                else:
                    # SAME TARGET: Force Re-Sync to DNN box to correct drift
                    # Only do this if we are not currently blind/moving
                    if current_time > last_move_end_time:
                        tgt_box = trackable_objects[best_id]['data'][0]
                        visual_tracker = _create_tracker()
                        if visual_tracker:
                            visual_tracker.init(frame, tuple(tgt_box))
            else:
                current_target_id = None
                visual_tracker = None

        # 4. Visual Tracking (Inter-frame)
        # We allow visual tracking to continue even if we are waiting for fresh DNN?
        # User said "redetect". Visual tracking uses *current* frame.
        # But if we just moved, KCF might be lost.
        # Let's Rely on MOT (fresh DNN) primarily after move.
        # If we interpret "redetect" strictly: Don't trust KCF immediately after move.
        # But KCF update needs continuous frames.
        
        target_box = None
        target_label = "Scanning"
        detected = False
        
        # Only predict if we have a target AND we are not "blind" from a move?
        # Actually, let's keep predictions running but gating the CONTROL is what matters.
        
        if current_target_id is not None and current_target_id in trackable_objects:
             obj = trackable_objects[current_target_id]
             target_label = obj['data'][1]
             
             if visual_tracker:
                 # CRITICAL: Do NOT run visual tracker if we are "Moving/Settling".
                 # The camera motion (optical flow) causes KCF to track the background shift.
                 if current_time < last_move_end_time:
                     # Blind Mode
                     detected = False
                     current_status = "Moving... (Tracking Paused)"
                 else:
                     ok, bbox = visual_tracker.update(frame)
                     if ok:
                         target_box = bbox
                         detected = True
             else:
                 target_box = obj['data'][0]
                 detected = True

        # 5. Robot Control
        # CRITICAL: Do NOT move if we haven't processed a fresh DNN result since the last move.
        # We need to ensure we are "Settled".
        # Check: Have we updated `trackable_objects` with post-move data?
        # `new_dnn_data` is only set if `ts > last_move_end_time`.
        # But `video_stream_loop` loops fast. `new_dnn_data` might be None this iteration.
        # We need to know if our *current understanding* (trackable_objects) is fresh.
        # The easiest way: Check `last_dnn_ts > last_move_end_time`.
        
        is_fresh_detection = (last_dnn_ts > last_move_end_time)
        
        if detected and target_box is not None and is_fresh_detection:
            last_seen_time = current_time
            head_recentered = False
            
            (x, y, w, h) = target_box
            cx = x + w/2
            
            # Smart Aiming: If tracking a 'person', aim for the Head (top ~20%)
            # Otherwise (face/cat), aim for Center.
            if target_label == 'person':
                cy = y + (h * 0.2)
            else:
                cy = y + (h / 2)
            
            current_status = f"Tracking ID {current_target_id} ({target_label})"
            
            err_x = (cx - frame_w / 2) / frame_w
            TARGET_Y = frame_h * 0.5
            err_y = (cy - TARGET_Y) / frame_h
            
            d_yaw = -err_x * np.deg2rad(FOV_X) * GAIN_YAW
            d_pitch = err_y * np.deg2rad(FOV_Y) * GAIN_PITCH
            
            # Wiggle
            if SERVER_STATE["wiggle_enabled"]:
                if not target_present or (current_time - last_detection_time > 5.0):
                    threading.Thread(target=robot.wiggle_antennas, daemon=True).start()
                    last_detection_time = current_time
            target_present = True
            
            # Move
            if current_time - last_command_time >= TUNING["command_interval"]:
                if not SERVER_STATE["paused"]:
                     MIN_MOVE = np.deg2rad(5.0)
                     if abs(d_yaw) > MIN_MOVE or abs(d_pitch) > MIN_MOVE:
                         duration = 1.0
                         # Log the move verification
                         ts_str = time.strftime("%H:%M:%S", time.localtime(current_time))
                         ms_str = f"{(current_time % 1):.3f}"[1:]
                         print(f"[{ts_str}{ms_str}] [MOVE] Validated Target ID {current_target_id} ({target_label}). Adjusting Head.")
                         robot.move_head(d_yaw, d_pitch, duration=duration)
                         last_command_time = current_time
                         
                         last_command_time = current_time
                         
                         # IMPORTANT: Mark when this move will end.
                         # Add safety buffer (0.5s) to ensure we are fully settled.
                         last_move_end_time = current_time + duration + 0.5
                         
                         # Force Rescan: Kill the visual tracker so we don't follow old ghosts.
                         # We often rely on this to be re-initialized by the next fresh DNN result.
                         visual_tracker = None
                         
                         # CRITICAL: Reset MOT Tracker
                         # We are moving the camera. Old centroids are now meaningless ghosts.
                         # Wipe memory to force fresh acquisition after move.
                         mot_tracker = SimpleTracker()
                         trackable_objects = {}
                         current_target_id = None 
                         head_recentered = True # Treat as a "reset" of sorts to prevent instant re-centering
        else:
             # Lost or Waiting for Fresh Data
             if not is_fresh_detection and current_target_id is not None:
                 current_status = "Waiting for post-move detection..."
             else:
                 target_present = False
        
        # Reset Logic
        if not detected and not head_recentered and (current_time - last_seen_time > RECENTER_TIMEOUT):
             if not SERVER_STATE["paused"]:
                 current_status = "Target Lost. Scanning..."
                 robot.recenter_head()
                 head_recentered = True
                 current_target_id = None

        # 6. Annotation (Draw ALL objects)
        annotated = frame.copy()
        
        # Draw all Candidates in Gray
        for obj_id, obj in trackable_objects.items():
            if obj_id == current_target_id: continue # Draw target last
            (ox, oy, ow, oh) = obj['data'][0]
            lbl = obj['data'][1]
            cv2.rectangle(annotated, (int(ox), int(oy)), (int(ox+ow), int(oy+oh)), (100, 100, 100), 1)
            # Increased Font Size 0.4 -> 0.8
            cv2.putText(annotated, f"ID {obj_id} {lbl}", (int(ox), int(oy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
 
        # Draw Target in Green/Orange
        if detected and target_box is not None:
             (tx, ty, tw, th) = target_box
             color = (0, 255, 0) if target_label == "face" else (0, 165, 255) # Green Face, Orange Cat
             cv2.rectangle(annotated, (int(tx), int(ty)), (int(tx+tw), int(ty+th)), color, 3)
             # Increased Font Size 0.6 -> 1.0
             cv2.putText(annotated, f"TARGET ID {current_target_id} {target_label}", (int(tx), int(ty)-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # MJPEG
        ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ret:
            latest_frame = buffer.tobytes()
            
        # FPS Sleep
        elapsed = time.perf_counter() - loop_start
        sleep_s = max(0.0, (1.0 / TUNING["stream_fps_cap"]) - elapsed)
        time.sleep(sleep_s)

def generate_mjpeg():
    while is_running:
        if latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.get("/")
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def get_status():
    return {
        "status": current_status,
        "paused": SERVER_STATE["paused"],
        "wiggle_enabled": SERVER_STATE["wiggle_enabled"],
        "candidates": LATEST_CANDIDATES,
        "pose": {
             "head_yaw": robot.current_yaw, # We might need to expose these from robot controller
             "head_pitch": robot.current_pitch,
             "head_roll": robot.current_roll,
             "body_yaw": robot.current_body_yaw,
             "antenna_left": robot.current_antenna_left,
             "antenna_right": robot.current_antenna_right
        }
    }

@app.post("/api/reset")
def api_reset():
    robot.recenter_head()
    return {"message": "Reset command sent"}

@app.post("/api/toggle_pause")
def toggle_pause():
    SERVER_STATE["paused"] = not SERVER_STATE["paused"]
    return {"status": "ok", "paused": SERVER_STATE["paused"]}

@app.post("/api/toggle_wiggle")
def toggle_wiggle():
    SERVER_STATE["wiggle_enabled"] = not SERVER_STATE["wiggle_enabled"]
    return {"status": "ok", "wiggle_enabled": SERVER_STATE["wiggle_enabled"]}

from pydantic import BaseModel

class ManualControlRequest(BaseModel):
    head_yaw: float
    head_pitch: float
    head_roll: float
    body_yaw: float
    antenna_left: float
    antenna_right: float

class MotorModeRequest(BaseModel):
    mode: str # stiff, limp, soft

@app.post("/api/motor_mode")
async def set_motor_mode(cmd: dict):
    mode = cmd.get("mode", "stiff")
    robot.set_motor_mode(mode)
    return {"status": "ok", "mode": mode}

@app.get("/api/tuning")
async def get_tuning():
    return TUNING

@app.post("/api/tuning")
async def set_tuning(new_tuning: dict):
    for k, v in new_tuning.items():
        if k in TUNING:
            TUNING[k] = float(v)
    return TUNING

@app.post("/api/manual_control")
def manual_control(req: ManualControlRequest):
    if not SERVER_STATE["paused"]:
        return {"status": "error", "message": "System must be paused for manual control."}
        
    robot.set_pose(
        head_yaw=req.head_yaw,
        head_pitch=req.head_pitch,
        head_roll=req.head_roll,
        body_yaw=req.body_yaw,
        antenna_left=req.antenna_left,
        antenna_right=req.antenna_right,
        duration=2.0 # Force 2s duration as requested
    )
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting server on port 8082...")
    robot.connect()
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=connection_monitor_loop, daemon=True).start()
    threading.Thread(target=video_stream_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8082)
