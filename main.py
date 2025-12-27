
import cv2
import threading
import time
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np

from robot_controller import RobotController
from detection_engine import DetectionEngine

try:
    cv2.setUseOptimized(True)
except Exception:
    pass

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Startup
    # threading.Thread(target=video_stream_loop, daemon=True).start() # Moved to main
    yield
    # Shutdown - nothing specific needed for now

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals
robot = RobotController()
detector = DetectionEngine()
current_status = "System Initializing..."
SERVER_STATE = {"paused": False}


# State
latest_frame = None
latest_camera_frame = None
latest_camera_lock = threading.Lock()

latest_dnn = {"detected": False, "box": None, "label": None, "conf": 0.0, "ts": 0.0}
latest_dnn_lock = threading.Lock()
is_running = True
last_detection_time = 0
DETECTION_COOLDOWN = 5.0 # Seconds
RECENTER_TIMEOUT = 3.0 # Seconds

# Tracking params
GAIN_YAW = 1.0  # Aggressive gain for fast centering
GAIN_PITCH = 1.0 # Aggressive gain for fast centering
FOV_X = 60 # degrees (approx webcam)
FOV_Y = 45 # degrees

# Performance tuning (Mac Mini M4)
STREAM_FPS_CAP = 15.0
JPEG_QUALITY = 70
DETECTION_INTERVAL_S = 0.25  # run DNN at most 4 Hz
COMMAND_INTERVAL_S = 0.10    # rate-limit robot commands to avoid thread churn

def _create_tracker():
    """Create a lightweight OpenCV tracker if available.

    MOSSE is very fast and good enough for short-term tracking between DNN updates.
    If trackers aren't available in this OpenCV build, return None.
    """
    candidates = [
        ("legacy", "TrackerMOSSE_create"),
        (None, "TrackerMOSSE_create"),
        ("legacy", "TrackerKCF_create"),
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
    """Run detector out-of-band so MJPEG streaming doesn't stall on DNN forward()."""
    global latest_camera_frame

def detection_loop():
    """Run detector out-of-band so MJPEG streaming doesn't stall on DNN forward()."""
    global latest_camera_frame

    while is_running:
        with latest_camera_lock:
            frame = None if latest_camera_frame is None else latest_camera_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue
        
        result = detector.detect(frame)
        if result is None:
            detected, box, label, conf = False, None, None, 0.0
        else:
            detected, box, label, conf = result
        now = time.time()
        with latest_dnn_lock:
            latest_dnn["detected"] = bool(detected)
            latest_dnn["box"] = box
            latest_dnn["label"] = label
            latest_dnn["conf"] = float(conf)
            latest_dnn["ts"] = now

        time.sleep(DETECTION_INTERVAL_S)

def connection_monitor_loop():
    """Background thread to handle robot connection retries without blocking video."""
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
    
    # PID / Control Loop Config
    FOV_X = 60.0 # deg (Reachy Camera)
    FOV_Y = 45.0
    
    # Tuned Gains (Softer for less twitchiness)
    GAIN_YAW = 0.7
    GAIN_PITCH = 0.7
    GAIN_YAW = 0.7
    GAIN_PITCH = 0.7
    GAIN_YAW = 0.7
    GAIN_PITCH = 0.7
    COMMAND_INTERVAL_S = 2.0 # Updates every 2.0s (Very slow, distinct moves)



    
    RECENTER_TIMEOUT = 2.0 # Seconds before resetting head
    DETECTION_COOLDOWN = 3.0 # Seconds between happy wiggles

    # Smoothing State (Exponential Moving Average)
    smooth_x, smooth_y, smooth_w, smooth_h = None, None, None, None
    ALPHA = 0.1 # Heavily smoothing detection (0.2 means 80% history)

    last_command_time = 0.0
    target_present = False

    tracker = None
    tracked_label = None
    last_dnn_refresh_ts = 0.0
    # last_command_time = 0.0 # This was moved above
    # target_present = False # This was moved above

    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    
    while is_running:
        loop_start = time.perf_counter()
        
        # Connection handling moved to connection_monitor_loop because it blocks!
        if not robot.is_connected:
            # Check pause state here too if needed, but primarily in logic
            pass

        # Connection handling moved to connection_monitor_loop because it blocks!
        
        # Check PAUSE removed from here (we want video to keep running)
        if SERVER_STATE["paused"]:
             current_status = "System PAUSED (Video Active)"
        
        # Get latest frame from robot (non-blocking now)
            
        # Get latest frame from robot (non-blocking now)

        frame = robot.get_latest_frame()
        if frame is not None:
            with latest_camera_lock:
                latest_camera_frame = frame
        
        current_time = time.time()

        detected = False
        box = None
        target_conf = 0.0

        # 1) Fast path: update tracker every frame
        if tracker is not None:
            try:
                ok, tracked = tracker.update(frame)
            except Exception:
                ok, tracked = False, None

            if ok and tracked is not None:
                (x, y, w, h) = tracked
                box = (int(x), int(y), int(w), int(h))
                target_label = tracked_label
                detected = True
            else:
                tracker = None
                tracked_label = None

        # 2) Refresh tracker from latest DNN result (computed in background)
        with latest_dnn_lock:
            dnn_snapshot = dict(latest_dnn)

        if dnn_snapshot["detected"] and dnn_snapshot["box"] is not None and (tracker is None or dnn_snapshot["ts"] > last_dnn_refresh_ts):
            last_dnn_refresh_ts = dnn_snapshot["ts"]
            detected = True
            box = dnn_snapshot["box"]
            target_label = dnn_snapshot["label"]
            target_conf = dnn_snapshot["conf"]

            new_tracker = _create_tracker()
            if new_tracker is not None:
                try:
                    new_tracker.init(frame, tuple(box))
                    tracker = new_tracker
                    tracked_label = target_label
                except Exception as e:
                    print(f"Tracker init failed: {e}") 
                    tracker = None
                    tracked_label = None
            if dnn_snapshot["detected"] and dnn_snapshot["box"] is not None and (tracker is None or dnn_snapshot["ts"] > last_dnn_refresh_ts):
                last_dnn_refresh_ts = dnn_snapshot["ts"]
                detected = True
                box = dnn_snapshot["box"]
                target_label = dnn_snapshot["label"]
                target_conf = dnn_snapshot["conf"]

                new_tracker = _create_tracker()
                if new_tracker is not None:
                    try:
                        new_tracker.init(frame, tuple(box))
                        tracker = new_tracker
                        tracked_label = target_label
                    except Exception:
                        tracker = None
                        tracked_label = None
            
            if detected and box is not None:
                last_seen_time = current_time
                head_recentered = False
                
                (x, y, w, h) = box
                
                # Apply EMA Smoothing
                if smooth_x is None:
                    smooth_x, smooth_y, smooth_w, smooth_h = x, y, w, h
                else:
                    smooth_x = ALPHA * x + (1 - ALPHA) * smooth_x
                    smooth_y = ALPHA * y + (1 - ALPHA) * smooth_y
                    smooth_w = ALPHA * w + (1 - ALPHA) * smooth_w
                    smooth_h = ALPHA * h + (1 - ALPHA) * smooth_h
                
                cx = smooth_x + smooth_w / 2
                cy = smooth_y + smooth_h / 2 # Track dead center

                label = (target_label or "target").upper()
                current_status = f"Tracking Active ({label}): Target at ({int(cx)}, {int(cy)})"
                
                frame_h, frame_w = frame.shape[:2]
                
                # Normalize error (-0.5 to 0.5)
                # Yaw: Error in X. If Object is Right (positive error), we need to Turn Left (positive Yaw) or Right?
                # Reachy: +Yaw is Left usually. 
                # If Object is at Right (x > w/2), error > 0. We need to turn Right (-Yaw).
                err_x = (cx - frame_w / 2) / frame_w
                err_y = (cy - frame_h / 2) / frame_h
                
                # Calculate angle steps (in radians)
                # -err_x because if object is Right (+), we usually reduce yaw (Right)
                # Pitch gain positive: err_y + (target low) -> Move Down (+)
                d_yaw = -err_x * np.deg2rad(FOV_X) * GAIN_YAW
                d_pitch = err_y * np.deg2rad(FOV_Y) * GAIN_PITCH
                
                if current_time - last_command_time >= COMMAND_INTERVAL_S:
                    # Only move if NOT PAUSED
                    if not SERVER_STATE["paused"]:
                         # Check for Minimum Motion (Deadband) to avoid micro-jitters
                         # 5 degrees ~ 0.087 rad. 
                         MIN_MOVE = np.deg2rad(5.0)
                         if abs(d_yaw) > MIN_MOVE or abs(d_pitch) > MIN_MOVE:
                             # Use longer duration for smoother "focusing" movements
                             # 1.5s duration with 2.0s interval
                             robot.move_head(d_yaw, d_pitch, duration=1.5)
                             last_command_time = current_time



                # Trigger actions on acquire (prevents constant retriggers, works for cat/person)
                # Also trigger periodically for liveness while tracking
                ANTENNA_INTERVAL = 5.0
                if (not target_present) or (current_time - last_detection_time > ANTENNA_INTERVAL):
                    # Only print if it's a fresh acquire, to avoid log spam
                    if not target_present:
                        print(f"Target acquired ({label})! Wiggling antennas.")
                        current_status = f"Target Acquired ({label})! Creating Joy..."
                    else:
                        # Periodic liveness
                        pass 
                        
                    threading.Thread(target=robot.wiggle_antennas, daemon=True).start()
                    last_detection_time = current_time

                target_present = True
            
            elif not head_recentered and (current_time - last_seen_time > RECENTER_TIMEOUT):
                print("Lost track of target. Recentering head.")
                current_status = "Target Lost. Scanning..."
                robot.recenter_head()
                head_recentered = True
                target_present = False
                smooth_x = None # Reset smoothing on loss
            
            # Encode for streaming
            annotated = frame
            if box is not None:
                annotated = frame.copy()
                (x, y, w, h) = box
                x2 = x + w
                y2 = y + h
                color = (0, 255, 0) if target_label == "cat" else (255, 128, 0)
                cv2.rectangle(annotated, (x, y), (x2, y2), color, 2)
                label_text = (target_label or "target").upper()
                if target_conf > 0.0:
                    label_text = f"{label_text}: {target_conf*100:.1f}%"
                y_label = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(annotated, label_text, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ret, buffer = cv2.imencode(
                '.jpg',
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if ret:
                latest_frame = buffer.tobytes()
        else:
            # If no frame, create a placeholder
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "No Video Feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            if ret:
                latest_frame = buffer.tobytes()
        
        # FPS cap to reduce CPU load from encoding/streaming on macOS
        elapsed = time.perf_counter() - loop_start
        sleep_s = max(0.0, (1.0 / STREAM_FPS_CAP) - elapsed)
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
    return {"status": current_status}

@app.post("/api/reset")
def api_reset():
    robot.recenter_head()
    robot.recenter_head()
    return {"message": "Reset command sent"}

@app.post("/api/toggle_pause")
def toggle_pause():
    SERVER_STATE["paused"] = not SERVER_STATE["paused"]
    state = "PAUSED" if SERVER_STATE["paused"] else "RESUMED"
    print(f"System {state}")
    return {"status": "ok", "paused": SERVER_STATE["paused"]}



if __name__ == "__main__":
    print("Starting server on port 8082...")

    # Initialize robot (and camera) in main thread first for MacOS compatibility
    robot.connect()
    
    # Start loop in background to keep fetching frames
    # (The camera object is created in main thread, reading from thread is usually ok but 
    # if it fails we might need to invert control)
    threading.Thread(target=detection_loop, daemon=True).start()
    threading.Thread(target=connection_monitor_loop, daemon=True).start()
    threading.Thread(target=video_stream_loop, daemon=True).start()

    try:
        # Disable the automatic loop start in lifespan since we do it manually above
        # But we need to keep lifespan for cleanup if needed, or we just remove the startup part there.
        uvicorn.run(app, host="0.0.0.0", port=8082)
    except Exception as e:
        print(f"Server crashed: {e}")
