
import cv2
import threading
import time
import os
import signal
import sys
import logging
import json
import subprocess
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import numpy as np
import webbrowser

from robot_controller import RobotController
from detection_engine import DetectionEngine
from simple_tracker import SimpleTracker
from voice_assistant import get_assistant
from llm_config import get_llm_config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress asyncio connection reset errors on Windows (harmless)
logging.getLogger('asyncio').setLevel(logging.WARNING)

try:
    cv2.setUseOptimized(True)
except Exception:
    pass

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

# Custom logging middleware to control verbosity
@app.middleware("http")
async def log_requests_middleware(request, call_next):
    """Control request logging based on endpoint and verbose setting."""
    global VERBOSE_LOGGING
    
    response = await call_next(request)
    
    # Skip logging for polling endpoints unless verbose logging is enabled
    quiet_endpoints = ["/status", "/api/voice/level", "/api/voice/status", "/api/voice/transcript", "/api/voice/response", "/api/voice/processing"]
    if request.url.path in quiet_endpoints and not VERBOSE_LOGGING:
        return response
    
    # Log other requests (or telemetry endpoints when verbose)
    status_code = response.status_code
    method = request.method
    path = request.url.path
    logger.info(f"{method:6s} {path:40s} {status_code}")
    
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals
robot = RobotController()
detector = DetectionEngine() # Now uses YOLOv8 (Output is list of dicts)
current_status = "System Initializing..."
SERVER_STATE = {
    "paused": True, 
    "wiggle_enabled": False
} # Default Paused, Wiggle OFF
server_instance = None  # Global uvicorn server instance for controlled shutdown
VERBOSE_LOGGING = False  # Control logging verbosity
LATEST_CANDIDATES = []
current_target_id = None
trackable_objects = {}

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
# Idle State
idle_start_time = 0.0
last_enforce_time = 0.0
idle_pose = None

# Tuning Parameters with Valid Ranges
# - detection_interval: 0.01-1.0 seconds (DNN execution frequency)
# - command_interval: 0.1-5.0 seconds (robot move frequency)
# - stream_fps_cap: 1-60 FPS (video stream frame rate limit)
# - min_score_threshold: 0-500 (minimum candidate score for tracking)
# - recenter_timeout: 0.5-10.0 seconds (idle recenter delay)
TUNING = {
    "detection_interval": 0.20, # Run DNN ~5Hz (YOLOv8n is fast)
    "command_interval": 1.2,    # 1.2s moves
    "stream_fps_cap": 60.0,
    "min_score_threshold": 250.0,
    "recenter_timeout": 2.0
}
TUNING_FILE = ".tuning_config.json"
WAKE_WORD_CONFIG_FILE = ".wake_word_config.json"
JPEG_QUALITY = 70

def _load_wake_word_settings():
    """Load wake word settings from file if exists."""
    if os.path.exists(WAKE_WORD_CONFIG_FILE):
        try:
            with open(WAKE_WORD_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded wake word settings from {WAKE_WORD_CONFIG_FILE}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load wake word settings: {e}")
    return None

def _save_wake_word_settings(config):
    """Save wake word settings to file."""
    try:
        with open(WAKE_WORD_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved wake word settings to {WAKE_WORD_CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to save wake word settings: {e}")
        return False

def _apply_wake_word_settings(voice_assistant):
    """Apply saved wake word settings to voice assistant"""
    saved_wake_word_config = _load_wake_word_settings()
    if saved_wake_word_config:
        voice_assistant.wake_word_enabled = saved_wake_word_config.get("enabled", True)
        voice_assistant.wake_word = saved_wake_word_config.get("wake_word", "hey_jarvis")
        voice_assistant.wake_word_threshold = saved_wake_word_config.get("threshold", 0.5)
        voice_assistant.wake_word_timeout = saved_wake_word_config.get("timeout", 5.0)
        
        # Restore audio input source preference with validation
        audio_input_source = saved_wake_word_config.get("audio_input_source")
        if audio_input_source:
            if audio_input_source == "sdk":
                # SDK is always available
                voice_assistant.set_audio_device("sdk")
                logger.info("Restored audio input: Reachy SDK")
            else:
                # Validate that the saved device is still available
                audio_device_id = saved_wake_word_config.get("audio_device_id")
                if audio_device_id is not None:
                    # Try to set the device - will return False if device no longer exists
                    if voice_assistant.set_audio_device(audio_device_id):
                        logger.info(f"Restored audio input device: {audio_device_id}")
                    else:
                        # Device no longer available, fall back to SDK
                        logger.warning(f"Saved audio device {audio_device_id} no longer available, reverting to SDK")
                        voice_assistant.set_audio_device("sdk")
                        # Update config to reflect the fallback
                        saved_wake_word_config["audio_input_source"] = "sdk"
                        saved_wake_word_config["audio_device_id"] = None
                        _save_wake_word_settings(saved_wake_word_config)
        else:
            # No audio source saved, default to SDK
            voice_assistant.set_audio_device("sdk")
            logger.info("No saved audio input, using SDK default")
        
        logger.info(f"Applied saved wake word settings: {saved_wake_word_config}")
        
        # Preload wake word model so status shows loaded on startup
        if voice_assistant.wake_word_enabled:
            try:
                voice_assistant._load_wake_word_model()
                logger.info("Wake word model preloaded on enable")
            except Exception as e:
                logger.warning(f"Could not preload wake word model: {e}")
        
        return True
    return False

def _load_tuning_settings():
    """Load tuning settings from file if exists."""
    global TUNING
    if os.path.exists(TUNING_FILE):
        try:
            with open(TUNING_FILE, 'r') as f:
                saved_tuning = json.load(f)
                TUNING.update(saved_tuning)
                logger.info(f"Loaded tuning settings from {TUNING_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load tuning settings: {e}")

def _save_tuning_settings():
    """Save tuning settings to file."""
    try:
        with open(TUNING_FILE, 'w') as f:
            json.dump(TUNING, f, indent=2)
        logger.info(f"Saved tuning settings to {TUNING_FILE}")
    except Exception as e:
        logger.error(f"Failed to save tuning settings: {e}")

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
    global latest_camera_frame, idle_start_time, last_enforce_time, idle_pose
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
    global current_target_id, trackable_objects
    
    last_seen_time = time.time()
    head_recentered = False
    
    # Tracking Logic
    mot_tracker = SimpleTracker(max_disappeared=5, max_distance=100)
    
    # Fast Visual Tracker (KCF) for the chosen target
    visual_tracker = None
    
    # Config
    FOV_X, FOV_Y = 60.0, 45.0
    # Increased Pitch Gain from 0.7 to 1.2 to force centering
    GAIN_YAW, GAIN_PITCH = 0.7, 1.2
    
    last_dnn_ts = 0.0
    last_move_end_time = 0.0
    
    # State
    target_present = False
    has_greeted_session = False
    last_detection_time = 0.0
    last_command_time = 0.0
    last_detection_time = 0.0
    last_command_time = 0.0
    last_state_update = 0.0
    last_seen_time = 0.0
    
    
    # RECENTER_TIMEOUT is now in TUNING
    
    global idle_start_time, last_enforce_time, idle_pose
    
    while is_running:
      try:
        loop_start = time.perf_counter()
        current_time = time.time()
        
        # Force State Update (5Hz) to keep UI fresh even when Paused
        if current_time - last_state_update > 0.2:
             try:
                 robot.update_head_pose()
                 last_state_update = current_time
             except: pass
        
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
            # Skip target switching if tracking is paused (e.g., during speech)
            if not robot._pause_tracking:
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
                         logger.info(f"SWITCH TARGET: {current_target_id} -> {best_id} | Candidates: {', '.join(debug_candidates)}")
                         current_target_id = best_id
                         tgt_box = trackable_objects[best_id]['data'][0]
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
        
        # Skip tracking if paused during speech
        if robot._pause_tracking:
            current_status = "Speaking... (Tracking Paused)"
            detected = False
            target_box = None
        elif current_target_id is not None and current_target_id in trackable_objects:
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
            # Wiggle
            if SERVER_STATE["wiggle_enabled"]:
                # Wiggle if new target (detected=True, prev=False) OR periodically
                if not target_present or (current_time - last_detection_time > 5.0):
                    threading.Thread(target=robot.wiggle_antennas, daemon=True).start()
                    last_detection_time = current_time
                    
                    # Play Sound (Only once per session)
                    if not has_greeted_session:
                        # Use TTS instead of static WAV
                        # Running in thread to avoid blocking video stream
                        try:
                            def say_hello():
                                assistant = get_assistant(robot)
                                if assistant:
                                    assistant.speak_text("Hello! I am Reachy. How can I help you today?")
                            
                            threading.Thread(target=say_hello, daemon=True).start()
                        except Exception:
                            pass

                        has_greeted_session = True
            
            target_present = True
            
            # Move
            if current_time - last_command_time >= TUNING["command_interval"]:
                if not SERVER_STATE["paused"]:
                     MIN_MOVE = np.deg2rad(5.0)
                     if abs(d_yaw) > MIN_MOVE or abs(d_pitch) > MIN_MOVE:
                         duration = 1.0
                         # Log the move verification
                         logger.info(f"[MOVE] Validated Target ID {current_target_id} ({target_label}). Adjusting Head.")
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
                         current_target_id = None 
                         head_recentered = True # Treat as a "reset" of sorts to prevent instant re-centering
                         
                         # Reset Idle State
                         idle_start_time = 0.0
                         idle_pose = None

                     else:
                         # IDLE STATE (Target is close enough)
                         
                         if idle_start_time == 0.0:
                             idle_start_time = current_time
                             last_enforce_time = current_time
                             # Capture current pose to hold
                             idle_pose = {
                                 "head_yaw": robot.current_yaw,
                                 "head_pitch": robot.current_pitch,
                                 "head_roll": 0.0, 
                                 "body_yaw": robot.current_body_yaw,
                                 "antenna_left": robot.current_antenna_left,
                                 "antenna_right": robot.current_antenna_right,
                                 "duration": 1.0
                             }
                        
                         # Enforce every 10s
                         if current_time - last_enforce_time > 10.0 and idle_pose:
                              print(f"[IDLE] Enforcing position on Target ID {current_target_id}")
                              robot.set_pose(**idle_pose)
                              last_enforce_time = current_time
        else:
             # Lost or Waiting for Fresh Data
             if not is_fresh_detection and current_target_id is not None:
                 current_status = "Waiting for post-move detection..."
             else:
                 target_present = False
                 target_present = False
        
        # Reset Logic
        if not detected and not head_recentered and (current_time - last_seen_time > TUNING["recenter_timeout"]):
             if not SERVER_STATE["paused"]:
                 current_status = "Target Lost. Scanning..."
                 robot.recenter_head()
                 head_recentered = True
                 current_target_id = None
                 has_greeted_session = False

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
        
      except Exception as e:
           logger.error(f"Video Loop Crash: {e}", exc_info=True)
           time.sleep(1.0) # Prevent tight loop crash

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
    # Gather current target metadata
    target_info = None
    if current_target_id is not None and current_target_id in trackable_objects:
        obj = trackable_objects[current_target_id]
        target_info = {
            "id": current_target_id,
            "label": obj.get('label', 'unknown'),
            "score": next((c['score'] for c in LATEST_CANDIDATES if c['id'] == current_target_id), 0)
        }
    
    return {
        "status": current_status,
        "paused": SERVER_STATE["paused"],
        "wiggle_enabled": SERVER_STATE["wiggle_enabled"],
        "current_target_id": current_target_id,
        "current_target": target_info,
        "candidates": LATEST_CANDIDATES,
        "volume": int(getattr(robot, 'audio_volume', 0.25) * 100),
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

@app.post("/api/emote")
async def api_emote(payload: dict):
    emotion = payload.get("emotion", "happy")
    print(f"[API] Emote request: {emotion}")
    robot.trigger_emotion(emotion)
    return {"status": "ok", "emotion": emotion}

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
    # Define valid ranges for each parameter
    RANGES = {
        "detection_interval": (0.01, 1.0),
        "command_interval": (0.1, 5.0),
        "stream_fps_cap": (1.0, 60.0),
        "min_score_threshold": (0.0, 500.0),
        "recenter_timeout": (0.5, 10.0)
    }
    
    updated = {}
    errors = []
    
    for k, v in new_tuning.items():
        if k in TUNING and k in RANGES:
            try:
                val = float(v)
                min_val, max_val = RANGES[k]
                
                # Clamp to valid range
                if val < min_val:
                    errors.append(f"{k}: {val} clamped to minimum {min_val}")
                    val = min_val
                elif val > max_val:
                    errors.append(f"{k}: {val} clamped to maximum {max_val}")
                    val = max_val
                
                TUNING[k] = val
                updated[k] = val
            except (ValueError, TypeError) as e:
                errors.append(f"{k}: invalid value '{v}' (must be numeric)")
        elif k not in TUNING:
            errors.append(f"{k}: unknown parameter (ignored)")
    
    # Save to file for persistence
    _save_tuning_settings()
    
    response = {"tuning": TUNING, "updated": updated}
    if errors:
        response["warnings"] = errors
        logger.warning(f"Tuning validation warnings: {errors}")
    
    logger.info(f"Tuning updated: {updated}")
    return response

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

@app.post("/api/restart")
def restart():
    """Restart the application by spawning a fresh process and shutting down current server."""
    global is_running, server_instance
    logger.info("Restart requested, restarting application...")
    is_running = False  # Stop loops

    main_path = Path(__file__).resolve()
    cwd = main_path.parent

    def restart_proc():
        try:
            # Give response time to flush back to client
            time.sleep(0.5)
            
            # Spawn new process BEFORE shutting down (so it can start while old one is exiting)
            logger.info(f"Spawning new process: {sys.executable} {main_path}")
            process = subprocess.Popen(
                [sys.executable, str(main_path)],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"New process started with PID {process.pid}")
            
            # Now shutdown uvicorn server gracefully to release port
            if server_instance:
                logger.info("Shutting down server gracefully...")
                server_instance.should_exit = True
                time.sleep(1.5)  # Wait for shutdown
            
            # Force exit to ensure clean restart
            logger.info("Exiting old process...")
            os._exit(0)
        except Exception as e:
            logger.error(f"Failed to restart: {e}", exc_info=True)
            os._exit(1)

    threading.Thread(target=restart_proc, daemon=True).start()
    return {"status": "ok", "message": "Restarting..."}

    return {"status": "restarting", "message": "System restarting..."}

@app.post("/api/shutdown")
def shutdown():
    global is_running
    is_running = False # Stop loops
    
    # Schedule kill
    def kill_proc():
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGINT)
        
    threading.Thread(target=kill_proc, daemon=True).start()
    return {"status": "shutting_down", "message": "System powering off..."}

# Voice Assistant State
VOICE_STATE = {
    "enabled": False,
    "listening": False,
    "last_transcript": "",
    "last_response": "",
    "conversation_history": [],
    "models": {},
    "error": None,
    "volume": 50,
}
voice_assistant = None

@app.post("/api/voice/enable")
def enable_voice():
    """Enable voice assistant"""
    global voice_assistant, VOICE_STATE
    try:
        if voice_assistant is None:
            voice_assistant = get_assistant(robot)
            
            # Apply saved wake word settings
            _apply_wake_word_settings(voice_assistant)
            logger.info("[INIT] Voice assistant created and configured")
        
        # Setup callbacks every time we enable (not just on first creation)
        # This ensures callbacks are registered even if voice_assistant was reused
        def on_wake_word():
            """Called when wake word is detected"""
            pass  # Typing indicator will be managed by is_processing flag
        
        def on_speech(text):
            VOICE_STATE["last_transcript"] = text
            VOICE_STATE["conversation_history"].append({"role": "user", "content": text})
        
        def on_response(text):
            VOICE_STATE["last_response"] = text
            VOICE_STATE["conversation_history"].append({"role": "assistant", "content": text})
        
        voice_assistant.on_wake_word_detected = on_wake_word
        voice_assistant.on_speech_detected = on_speech
        voice_assistant.on_response_ready = on_response
        logger.info("[INIT] Voice assistant callbacks registered")
        
        voice_assistant.start_listening()
        # Snapshot model info (may update after lazy load)
        try:
            VOICE_STATE["models"] = voice_assistant.get_model_info()
            VOICE_STATE["error"] = VOICE_STATE["models"].get("error")
        except Exception:
            pass
        VOICE_STATE["enabled"] = True
        VOICE_STATE["listening"] = True
        logger.info("Voice assistant enabled")
        return {"status": "ok", "voice_enabled": True}
    except Exception as e:
        logger.error(f"Failed to enable voice assistant: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post("/api/voice/disable")
def disable_voice():
    """Disable voice assistant"""
    global voice_assistant, VOICE_STATE
    if voice_assistant:
        voice_assistant.stop_listening()
        VOICE_STATE["enabled"] = False
        VOICE_STATE["listening"] = False
        logger.info("Voice assistant disabled")
    return {"status": "ok", "voice_enabled": False}

@app.get("/api/voice/status")
def voice_status():
    """Get voice assistant status"""
    return VOICE_STATE

@app.post("/api/voice/volume")
def set_voice_volume(data: dict):
    """Set speaker volume percentage (0-100)."""
    global VOICE_STATE, robot
    try:
        vol = int(data.get("volume", 50))
        vol = max(0, min(100, vol))
        VOICE_STATE["volume"] = vol
        # Apply to robot controller (0.0 - 1.0)
        try:
            robot.set_audio_volume(vol / 100.0)
        except Exception:
            pass
        return {"status": "ok", "volume": vol}
    except Exception as e:
        logger.error(f"Volume set error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/voice/text_command")
async def text_command(data: dict):
    """Process a text command through the LLM"""
    global voice_assistant
    text = data.get("text", "")
    
    if not text:
        return {"status": "error", "message": "No text provided"}
    
    try:
        if voice_assistant is None:
            voice_assistant = get_assistant(robot)
            _apply_wake_word_settings(voice_assistant)
        
        response = voice_assistant.process_text_command(text)
        
        # Update history
        VOICE_STATE["last_transcript"] = text
        VOICE_STATE["last_response"] = response
        VOICE_STATE["conversation_history"].append({"role": "user", "content": text})
        VOICE_STATE["conversation_history"].append({"role": "assistant", "content": response})
        
        # Speak asynchronously in background so response is returned immediately
        if data.get("speak", False):
            logger.info("[VOICE] Speak requested: sending TTS to robot in background")
            import threading
            threading.Thread(target=lambda: voice_assistant.speak_text(response), daemon=True).start()
        
        return {"status": "ok", "response": response}
    except Exception as e:
        logger.error(f"Text command error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post("/api/voice/listen_now")
async def listen_now():
    """Manual trigger: Start listening for a voice command RIGHT NOW
    
    This will:
    1. Play 'excited' emotion to indicate listening started
    2. Capture speech for up to 10 seconds
    3. Transcribe and respond
    4. Return to idle
    """
    global voice_assistant
    
    try:
        if voice_assistant is None or not voice_assistant.running:
            return {"status": "error", "message": "Voice assistant not running"}
        
        # Trigger manual listening mode
        voice_assistant.manual_listen_trigger()
        
        return {"status": "ok", "message": "Listening started - speak now!"}
    except Exception as e:
        logger.error(f"Manual listen trigger error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post("/api/voice/speak")
async def speak_text(data: dict):
    """Speak text directly via TTS"""
    global voice_assistant
    text = data.get("text", "")
    
    if not text:
        return {"status": "error", "message": "No text provided"}
    
    try:
        if voice_assistant is None:
            voice_assistant = get_assistant(robot)
            _apply_wake_word_settings(voice_assistant)
        
        voice_assistant.speak_text(text)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"TTS error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/voice/stop")
async def stop_speech():
    """Stop currently playing speech"""
    global voice_assistant
    try:
        if voice_assistant:
            voice_assistant.stop_speech()
            return {"status": "ok"}
        return {"status": "error", "message": "Voice assistant not initialized"}
    except Exception as e:
        logger.error(f"Stop speech error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post("/api/voice/wake-word/configure")
async def configure_wake_word(data: dict):
    """Configure wake word detection settings"""
    global voice_assistant
    
    try:
        if voice_assistant is None:
            voice_assistant = get_assistant(robot)
            _apply_wake_word_settings(voice_assistant)
        
        # Update settings
        if "enabled" in data:
            voice_assistant.wake_word_enabled = bool(data["enabled"])
        if "wake_word" in data:
            # Validate wake word model name - check both built-in and custom models
            import os
            import glob
            
            # Built-in models
            valid_models = ["alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy"]
            
            # Custom models in models/openwakeword directory
            custom_model_dir = os.path.join(os.path.dirname(__file__), "models", "openwakeword")
            if os.path.exists(custom_model_dir):
                custom_models = [os.path.splitext(os.path.basename(f))[0] 
                                for f in glob.glob(os.path.join(custom_model_dir, "*.onnx"))]
                valid_models.extend(custom_models)
            
            if data["wake_word"] in valid_models:
                voice_assistant.wake_word = data["wake_word"]
                # Reset model to load new wake word
                voice_assistant._wake_word_model = None
            else:
                return {"status": "error", "message": f"Invalid wake word. Valid options: {valid_models}"}
        if "threshold" in data:
            threshold = float(data["threshold"])
            if 0.0 <= threshold <= 1.0:
                voice_assistant.wake_word_threshold = threshold
            else:
                return {"status": "error", "message": "Threshold must be between 0.0 and 1.0"}
        if "timeout" in data:
            timeout = float(data["timeout"])
            if timeout > 0:
                voice_assistant.wake_word_timeout = timeout
            else:
                return {"status": "error", "message": "Timeout must be positive"}
        if "audio_device_id" in data:
            device_selection = data["audio_device_id"]
            # Handle both "sdk" string and numeric device IDs
            if device_selection == "sdk":
                device_id_to_set = "sdk"
            else:
                try:
                    device_id_to_set = int(device_selection)
                except (ValueError, TypeError):
                    return {"status": "error", "message": f"Invalid audio device ID: {device_selection}"}
            
            if voice_assistant.set_audio_device(device_id_to_set):
                logger.info(f"Audio input source updated to {device_id_to_set}")
            else:
                return {"status": "error", "message": f"Invalid audio device selection: {device_id_to_set}"}
        
        # Save settings to file
        config = {
            "enabled": voice_assistant.wake_word_enabled,
            "wake_word": voice_assistant.wake_word,
            "threshold": voice_assistant.wake_word_threshold,
            "timeout": voice_assistant.wake_word_timeout,
            "audio_input_source": voice_assistant.audio_input_source,
            "audio_device_id": voice_assistant.audio_device_id,
            "audio_device_name": voice_assistant.audio_device_name
        }
        _save_wake_word_settings(config)
        
        return {
            "status": "ok",
            "config": config
        }
    except Exception as e:
        logger.error(f"Wake word configuration error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/voice/audio-devices")
async def get_audio_devices():
    """Get list of available audio input devices"""
    global voice_assistant
    
    try:
        if voice_assistant is None:
            voice_assistant = get_assistant(robot)
            _apply_wake_word_settings(voice_assistant)
        
        devices = voice_assistant.get_available_audio_devices()
        return {
            "status": "ok",
            "devices": devices,
            "selected_audio_input_source": voice_assistant.audio_input_source,
            "selected_device_id": voice_assistant.audio_device_id,
            "selected_device_name": voice_assistant.audio_device_name
        }
    except Exception as e:
        logger.error(f"Error getting audio devices: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.get("/api/voice/wake-word/status")
async def get_wake_word_status():
    """Get current wake word detection configuration"""
    global voice_assistant
    
    try:
        import os
        import glob
        
        # Get available models
        valid_models = ["alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy"]
        custom_model_dir = os.path.join(os.path.dirname(__file__), "models", "openwakeword")
        if os.path.exists(custom_model_dir):
            custom_models = [os.path.splitext(os.path.basename(f))[0] 
                            for f in glob.glob(os.path.join(custom_model_dir, "*.onnx"))]
            valid_models.extend(custom_models)
        
        # Load saved settings if available
        saved_config = _load_wake_word_settings()
        
        if voice_assistant is None:
            # Return saved settings or defaults
            if saved_config:
                return {
                    "enabled": saved_config.get("enabled", True),
                    "wake_word": saved_config.get("wake_word", "hey_jarvis"),
                    "threshold": saved_config.get("threshold", 0.5),
                    "timeout": saved_config.get("timeout", 5.0),
                    "audio_input_source": saved_config.get("audio_input_source", "sdk"),
                    "audio_device_id": saved_config.get("audio_device_id"),
                    "available_models": valid_models,
                    "model_loaded": False
                }
            else:
                return {
                    "enabled": True,
                    "wake_word": "hey_jarvis",
                    "threshold": 0.5,
                    "timeout": 5.0,
                    "audio_input_source": "sdk",
                    "audio_device_id": None,
                    "available_models": valid_models,
                    "model_loaded": False
                }
        
        return {
            "enabled": voice_assistant.wake_word_enabled,
            "wake_word": voice_assistant.wake_word,
            "threshold": voice_assistant.wake_word_threshold,
            "timeout": voice_assistant.wake_word_timeout,
            "audio_input_source": voice_assistant.audio_input_source,
            "audio_device_id": voice_assistant.audio_device_id,
            "audio_device_name": voice_assistant.audio_device_name,
            "available_models": valid_models,
            "model_loaded": voice_assistant._wake_word_model is not None
        }
    except Exception as e:
        logger.error(f"Wake word status error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/api/voice/level")
async def get_voice_level():
    """Return latest microphone level telemetry"""
    global voice_assistant
    try:
        if voice_assistant is None:
            return {"rms": 0.0, "db": -120.0}
        return {
            "rms": voice_assistant.last_rms,
            "db": voice_assistant.last_db
        }
    except Exception as e:
        logger.error(f"Voice level error: {e}", exc_info=True)
        return {"rms": 0.0, "db": -120.0}


@app.post("/api/voice/debug")
async def set_debug_mode(request_data: dict):
    """Enable/disable debug audio logging"""
    global voice_assistant
    try:
        enabled = request_data.get("enabled", False)
        if voice_assistant:
            voice_assistant.debug_audio_enabled = enabled
            logger.info(f"Debug audio logging: {'ENABLED' if enabled else 'DISABLED'}")
            return {"status": "ok", "debug_enabled": enabled}
        return {"status": "error", "message": "Voice assistant not initialized"}
    except Exception as e:
        logger.error(f"Debug mode error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/api/voice/transcript")
async def get_latest_transcript():
    """Get the latest transcribed text from speech input"""
    global VOICE_STATE
    try:
        text = VOICE_STATE.get("last_transcript", "")
        if text:
            # Return and clear so it's only shown once
            last = text
            VOICE_STATE["last_transcript"] = ""
            return {"text": last, "has_new": True}
        return {"text": "", "has_new": False}
    except Exception as e:
        logger.error(f"Transcript fetch error: {e}", exc_info=True)
        return {"text": "", "has_new": False}

@app.get("/api/voice/response")
async def get_latest_response():
    """Get the latest assistant response"""
    global VOICE_STATE
    try:
        text = VOICE_STATE.get("last_response", "")
        if text:
            # Return and clear so it's only shown once
            last = text
            VOICE_STATE["last_response"] = ""
            return {"text": last, "has_new": True}
        return {"text": "", "has_new": False}
    except Exception as e:
        logger.error(f"Response fetch error: {e}", exc_info=True)
        return {"text": "", "has_new": False}

@app.get("/api/voice/processing")
async def get_processing_status():
    """Check if voice assistant is currently processing"""
    global voice_assistant
    try:
        if voice_assistant is None:
            return {"processing": False}
        return {"processing": voice_assistant.is_processing}
    except Exception as e:
        logger.error(f"Processing status error: {e}", exc_info=True)
        return {"processing": False}


# ══════════════════════════════════════
# LLM CONFIGURATION ROUTES
# ══════════════════════════════════════

@app.get("/api/llm/models/local")
async def get_local_models():
    """Get list of available local models."""
    try:
        llm_config = get_llm_config_manager()
        models = llm_config.get_local_models()
        current = llm_config.get_current_config()
        # Mark current model
        for m in models:
            if m["id"] == current.get("model_id"):
                m["current"] = True
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching local models: {e}")
        return {"models": [], "error": str(e)}


@app.get("/api/llm/models/ollama")
async def get_ollama_models(endpoint: str = "http://localhost:11434"):
    """Get list of OLLAMA models from endpoint."""
    try:
        llm_config = get_llm_config_manager()
        models, error = llm_config.get_ollama_models(endpoint)
        if error:
            return {"models": [], "error": error}
        return {"models": models}
    except Exception as e:
        logger.error(f"Error fetching OLLAMA models: {e}")
        return {"models": [], "error": str(e)}


@app.post("/api/llm/models/openai")
async def get_openai_models(data: dict):
    """Get list of OpenAI models and pricing."""
    try:
        api_key = data.get("api_key")
        
        # If api_key is None, use the stored key from config
        if api_key is None:
            llm_config = get_llm_config_manager()
            api_key = llm_config.get_openai_key()
        
        if not api_key:
            return {"models": [], "error": "API key required"}
        
        llm_config = get_llm_config_manager()
        models, pricing = llm_config.get_openai_models(api_key)
        
        # Format pricing for display with category
        pricing_display = [
            {
                "model": p["name"],
                "category": p.get("category", "Standard"),
                "input_price": f"${p['input']:.4f}",
                "output_price": f"${p['output']:.2f}",
                "cost_per_1m": f"${p['input'] * 1000:.2f} / ${p['output'] * 1000:.2f}"
            }
            for p in pricing
        ]
        
        return {"models": models, "pricing": pricing_display}
    except Exception as e:
        logger.error(f"Error fetching OpenAI models: {e}")
        return {"models": [], "error": str(e)}


@app.post("/api/llm/validate-key")
async def validate_api_key(data: dict):
    """Validate API key for a provider."""
    try:
        provider = data.get("provider", "openai")
        api_key = data.get("api_key")
        
        if not api_key:
            return {"valid": False, "error": "API key required"}
        
        if provider == "openai":
            llm_config = get_llm_config_manager()
            valid, msg = llm_config.validate_openai_key(api_key)
            return {"valid": valid, "error": msg if not valid else None}
        
        return {"valid": False, "error": f"Provider {provider} validation not implemented"}
    except Exception as e:
        logger.error(f"Key validation error: {e}")
        return {"valid": False, "error": str(e)}


@app.post("/api/llm/config")
async def save_llm_config(data: dict):
    """Save LLM provider configuration."""
    try:
        provider = data.get("provider")
        config = data.get("config", {})
        
        if not provider:
            return {"success": False, "error": "Provider required"}
        
        llm_config = get_llm_config_manager()
        success, message = llm_config.set_provider_config(provider, config)
        
        if success:
            logger.info(f"LLM config updated: provider={provider}")
            return {"success": True, "message": message}
        else:
            return {"success": False, "error": message}
    except Exception as e:
        logger.error(f"Error saving LLM config: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/llm/status")
async def get_llm_status():
    """Get current LLM provider and model info."""
    try:
        llm_config = get_llm_config_manager()
        provider = llm_config.get_current_provider()
        config = llm_config.get_current_config()
        
        # Check if OpenAI API key is saved (without exposing it)
        has_api_key = "api_key" in config and config.get("api_key") is not None
        
        safe_config = {k: v for k, v in config.items() if k != "api_key"}
        if has_api_key and provider == "openai":
            safe_config["has_api_key"] = True
        
        return {
            "provider": provider,
            "config": safe_config
        }
    except Exception as e:
        logger.error(f"Error fetching LLM status: {e}")
        return {"error": str(e)}


@app.post("/api/settings/verbose")
async def set_verbose_logging(data: dict):
    """Enable/disable verbose logging."""
    global VERBOSE_LOGGING
    try:
        VERBOSE_LOGGING = data.get("verbose", False)
        logger.info(f"Verbose logging: {'enabled' if VERBOSE_LOGGING else 'disabled'}")
        return {"success": True, "verbose": VERBOSE_LOGGING}
    except Exception as e:
        logger.error(f"Error setting verbose logging: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    logger.info("Starting server on port 8082...")
    logger.info("Use http://localhost:8082/ to access the web interface.")
    
    # Load saved tuning settings
    _load_tuning_settings()
    
    webbrowser.open("http://localhost:8082/")    
    robot.connect()
    
    # Preload AI models in background to avoid delay on first chat
    try:
        logger.info("Triggering background model load...")
        va = get_assistant(robot)
        va.preload_models()
    except Exception as e:
        logger.error(f"Failed to start model preload: {e}")

    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=connection_monitor_loop, daemon=True).start()
    threading.Thread(target=video_stream_loop, daemon=True).start()
    
    # Use Server class for controlled shutdown
    config = uvicorn.Config(app, host="0.0.0.0", port=8082, access_log=False)
    server_instance = uvicorn.Server(config)
    server_instance.run()