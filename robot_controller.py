import numpy as np
import time
import requests
import cv2
import threading
import platform
import queue
from reachy_sdk_shim import ReachyMini, create_head_pose

class RobotController:
    def __init__(self, host='localhost'):
        self.host = host
        self.base_url = None
        self.camera = None
        self._is_connected_to_api = False
        
        # Command Queue for serialized execution
        self.command_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self._worker_thread.start()
        
        # Camera State
        self._latest_camera_frame = None
        self._camera_lock = threading.Lock()
        self._camera_thread = None
        
        # Current head state (cache)
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_body_yaw = 0.0

    @property
    def is_connected(self):
        return self._is_connected_to_api

    def disconnect(self):
        self._is_connected_to_api = False
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None

    def connect(self, silent=False):
        # If already connected, do check
        if self.is_connected and self.camera is not None and self.camera.isOpened():
            return

        # Prioritize 127.0.0.1 to avoid localhost resolution issues on Mac
        possible_hosts = ['127.0.0.1', 'localhost', 'reachy.local', self.host]
        # Deduplicate
        hosts_to_try = []
        for h in possible_hosts:
            if h and h not in hosts_to_try:
                hosts_to_try.append(h)
        
        # 1. Connect to API
        api_success = False
        for host in hosts_to_try:
            url = f"http://{host}:8000"
            try:
                # Health check or Status
                resp = requests.get(f"{url}/api/daemon/status", timeout=2)
                if resp.status_code == 200:
                    print(f"Successfully connected to API at {url}!", flush=True)
                    self.host = host
                    self.base_url = url
                    api_success = True
                    
                    # Try to force motors ON (compliance false)
                    try:
                         # Shotgun turn on
                         requests.post(f"{url}/api/turn_on", json={}, timeout=1)
                         requests.post(f"{url}/api/compliant", json={"compliant": False}, timeout=1)
                    except:
                        pass
                        
                    # Initial state fetch
                    self.update_head_pose()
                    break
            except Exception as e:
                # if not silent:
                #     print(f"Failed to connect to API at {host}: {e}")
                pass
        
        if not api_success:
            if not silent:
                print("API Connection failed.")
        
        # 2. Connect to Camera
        if self.camera is None or not self.camera.isOpened():
            try:
                for idx in [0, 1]:
                    if platform.system() == 'Windows':
                        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    else:
                        cap = cv2.VideoCapture(idx) # Auto-detect (AVFoundation on Mac, V4L2 on Linux)
                        
                    if not cap.isOpened():
                         cap = cv2.VideoCapture(idx) # Fallback to auto
                    
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            self.camera = cap
                            
                            # Force hardware settings for better low-light performance
                            try:
                                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.camera.set(cv2.CAP_PROP_FPS, 15) 
                                self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                            except:
                                pass
                            
                            print(f"Camera initialized on index {idx}.")
                            
                            if self._camera_thread is None or not self._camera_thread.is_alive():
                                self._camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
                                self._camera_thread.start()
                            break
                        else:
                            cap.release()
            except Exception as e:
                print(f"Camera error: {e}")

        if self.camera is None:
            if not silent:
                print("No working camera found.")
        
        self._is_connected_to_api = api_success

        

    def _camera_worker(self):
        """Background thread to read frames as fast as possible (latencyless)."""
        while self.camera and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Fix color space issues (Mac often returns BGRA or YUYV)
                    if frame.ndim == 3:
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        elif frame.shape[2] == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                    
                    with self._camera_lock:
                        self._latest_camera_frame = frame
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Camera worker exception: {e}")
                time.sleep(0.1)

    def get_latest_frame(self):
        with self._camera_lock:
            if self._latest_camera_frame is None:
                return None
            return self._latest_camera_frame.copy()
            
    def _command_worker(self):
        """Background thread to process movement commands sequentially."""
        while True:
            try:
                func, args = self.command_queue.get()
                try:
                    func(*args)
                except Exception as e:
                    print(f"Error in command worker: {e}", flush=True)
                finally:
                    self.command_queue.task_done()
            except Exception:
                pass

    def move_head(self, d_yaw, d_pitch, duration=0.5):
        """Enqueue a relative head movement command."""
        # Empty queue if it's getting backed up to ensure fresh commands
        if self.command_queue.qsize() > 2:
            try:
                while not self.command_queue.empty():
                    self.command_queue.get_nowait()
                    self.command_queue.task_done()
            except queue.Empty:
                pass
                
        self.command_queue.put((self._move_head_sync, (d_yaw, d_pitch, duration)))

    def _move_head_sync(self, d_yaw, d_pitch, duration=0.5):
        """
        Move head relative to current position.
        Also moves body if head turns too far.
        """
        if not self.base_url:
            self.current_yaw += d_yaw
        try:
            # Update current first
            self.update_head_pose()
            
            target_head_yaw = self.current_yaw + d_yaw
            target_pitch = self.current_pitch + d_pitch
            target_body_yaw = self.current_body_yaw

            # Zone-Based Tracking Logic (40% - 20% - 40%)
            # FOV_X is approx 60 degrees.
            # Center zone (20%) is +/- 10% of FOV = +/- 6 degrees = +/- 0.1 radians.
            
            ZONE_THRESHOLD = np.deg2rad(6.0) # +/- 6 degrees
            
            new_body_yaw = target_body_yaw
            new_head_yaw = target_head_yaw # Default: head stays relative to body
            
            # Use 'd_yaw' (the error) to determine zone, rather than absolute position
            # d_yaw is the offset from center.
            
            if abs(d_yaw) > ZONE_THRESHOLD:
                # SIDE ZONES (40% Left or Right) -> Move BODY
                # Eliminate the error using the body.
                
                # We want to change the body angle by 'd_yaw' amount to center it.
                # If d_yaw is Negative (Target Right), we want to turn Right (Negative).
                # So we ADD d_yaw.
                new_body_yaw = target_body_yaw + d_yaw
                
                # Fix for "Head stays in same position" (Counter-Rotation)
                # If SDK Head Yaw is Global (Base Frame), setting it to 0 keeps it at Base Front.
                # To make it turn WITH the body, we must set Head Yaw to the SAME angle as Body Yaw.
                # Body moves to 'new_body_yaw'. Head should look at 'new_body_yaw'.
                new_head_yaw = new_body_yaw 
                
            else:
                # CENTER ZONE (20%) -> Move HEAD
                # Fine tuning. Body stays put.
                target_head_yaw = self.current_yaw + d_yaw # Standard head update
                new_head_yaw = target_head_yaw
                new_body_yaw = target_body_yaw
            
            # Apply changes
            target_body_yaw = new_body_yaw
            target_head_yaw = new_head_yaw

            # Clamp limits
            target_head_yaw = np.clip(target_head_yaw, -1.0, 1.0)     # Head limits
            target_pitch = np.clip(target_pitch, -0.8, 0.2)
            target_body_yaw = np.clip(target_body_yaw, -2.5, 2.5)
    
            # USE NEW SDK SHIM
            with ReachyMini(self.host) as mini:
                # DEBUG LOGGING (User Requested)
                print(f"[DEBUG] Zone Logic: d_yaw_err={d_yaw:.3f} | Zone={'SIDE' if abs(d_yaw) > ZONE_THRESHOLD else 'CENTER'}", flush=True)
                print(f"[DEBUG] Target: HeadYaw={target_head_yaw:.3f} BodyYaw={target_body_yaw:.3f} Pitch={target_pitch:.3f}", flush=True)
                
                mini.goto_target(
                    head=create_head_pose(
                        roll=self.current_roll,
                        pitch=target_pitch,
                        yaw=target_head_yaw,
                        x=0, y=0, z=0
                    ),
                    body_yaw=target_body_yaw,
                    duration=duration,
                    method="minjerk"
                )
                
                # UPDATE STATE (Dead Reckoning)
                # This fixes the "snapping" bug by remembering where we told it to go
                self.current_body_yaw = target_body_yaw
                self.current_pitch = target_pitch
                # self.current_yaw keeps track of head relative to body? 
                # In this logic current_yaw is treated as head position.
                self.current_yaw = target_head_yaw
        except:
            pass
            
    def update_head_pose(self):
        """Fetch current head pose from API (Stubbed)."""
        pass

    def recenter_head(self):
        self.command_queue.put((self._recenter_head_sync, ()))

    def _recenter_head_sync(self):
        """Moves head to default center position (0, 0, 0)."""
        # Always reset internal state to 0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_body_yaw = 0.0

        if not self.base_url:
            return
        
        try:
            with ReachyMini(self.host) as mini:
                mini.goto_target(
                    head=create_head_pose(roll=0, pitch=0, yaw=0, x=0, y=0, z=0),
                    body_yaw=0.0,
                    antennas=np.zeros(2), 
                    duration=3.0, # Slow reset
                    method="minjerk"
                )
        except:
            pass

    def wiggle_antennas(self):
        self.command_queue.put((self._wiggle_antennas_sync, ()))

    def _wiggle_antennas_sync(self):
        """Wiggles antennas to show happiness/detection."""
        if not self.base_url:
            return
            
        def move_antennas(l_deg, r_deg, duration):
            try:
                with ReachyMini(self.host) as mini:
                    mini.goto_target(
                        antennas=np.deg2rad([l_deg, r_deg]),
                        duration=duration,
                        method="minjerk"
                    )
            except Exception as e:
                print(f"DEBUG: Wiggle failed: {e}", flush=True)

        try:
            # Slower, gentler wiggle
            for _ in range(2):
                move_antennas(30, -30, 0.4)
                time.sleep(0.4)
                move_antennas(-10, 10, 0.4)
                time.sleep(0.4)
            move_antennas(0, 0, 0.6)
        except:
            pass

    def reset_antennas(self):
        if self.base_url:
            try:
                requests.post(f"{self.base_url}/api/move/goto", json={
                    "antennas": [0, 0],
                    "duration": 1.0
                }, timeout=1)
            except:
                pass
