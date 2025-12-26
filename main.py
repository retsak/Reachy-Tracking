
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

# State
latest_frame = None
is_running = True
last_detection_time = 0
DETECTION_COOLDOWN = 5.0 # Seconds
RECENTER_TIMEOUT = 3.0 # Seconds

# Tracking params
GAIN_YAW = 1.0  # Aggressive gain for fast centering
GAIN_PITCH = 1.0 # Aggressive gain for fast centering
FOV_X = 60 # degrees (approx webcam)
FOV_Y = 45 # degrees

def video_stream_loop():
    global latest_frame, last_detection_time
    
    last_seen_time = time.time()
    head_recentered = False
    
    # Try to connect initial
    robot.connect()
    
    while is_running:
        if not robot.is_connected:
            # Try reconnecting every few seconds
            time.sleep(2)
            robot.connect(silent=True)
            
        frame = robot.get_latest_frame()
        if frame is not None:
            # Run detection
            detected, box, annotated = detector.detect(frame)
            current_time = time.time()
            
            if detected and box is not None:
                last_seen_time = current_time
                head_recentered = False
                
                (x, y, w, h) = box
                cx = x + w / 2
                cy = y + h * 0.2 # Track face (top 20%)
                
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
                
                robot.move_head(d_yaw, d_pitch)

                if current_time - last_detection_time > DETECTION_COOLDOWN:
                    print(f"Cat detected at ({cx:.1f}, {cy:.1f})! Wiggling antennas.")
                    threading.Thread(target=robot.wiggle_antennas).start()
                    last_detection_time = current_time
            
            elif not head_recentered and (current_time - last_seen_time > RECENTER_TIMEOUT):
                print("Lost track of cat. Recentering head.")
                robot.recenter_head()
                head_recentered = True
            
            # Encode for streaming
            ret, buffer = cv2.imencode('.jpg', annotated)
            if ret:
                latest_frame = buffer.tobytes()
        else:
            # If no frame, create a placeholder
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "No Video Feed", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            if ret:
                latest_frame = buffer.tobytes()
        
        time.sleep(0.03) # ~30 FPS Cap

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


if __name__ == "__main__":
    print("Starting server on port 8082...")
    try:
    # Initialize robot (and camera) in main thread first for MacOS compatibility
    robot.connect()
    
    # Start loop in background to keep fetching frames
    # (The camera object is created in main thread, reading from thread is usually ok but 
    # if it fails we might need to invert control)
    threading.Thread(target=video_stream_loop, daemon=True).start()

    try:
        # Disable the automatic loop start in lifespan since we do it manually above
        # But we need to keep lifespan for cleanup if needed, or we just remove the startup part there.
        uvicorn.run(app, host="0.0.0.0", port=8082)
    except Exception as e:
        print(f"Server crashed: {e}")
