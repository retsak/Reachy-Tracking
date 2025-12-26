
import numpy as np
import time
import requests
import cv2
import threading
import platform

class RobotController:
    def __init__(self, host='localhost'):
        self.host = host
        self.base_url = None
        self.camera = None
        self._is_connected_to_api = False
        self._is_moving = False
        
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

        possible_hosts = ['localhost', '127.0.0.1', 'reachy.local', self.host]
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
                if not silent:
                    print(f"Attempting to connect to Reachy API at {url}...")
                
                # Health check or Status
                resp = requests.get(f"{url}/api/daemon/status", timeout=2)
                if resp.status_code == 200:
                    print(f"Successfully connected to API at {url}!")
                    self.host = host
                    self.base_url = url
                    api_success = True
                    # Initial state fetch
                    self.update_head_pose()
                    break
            except Exception as e:
                if not silent:
                    print(f"Failed to connect to API at {host}: {e}")
        
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
                            print(f"Camera initialized on index {idx}.")
                            break
                        else:
                            cap.release()
            except Exception as e:
                print(f"Camera error: {e}")

        if self.camera is None:
            if not silent:
                print("No working camera found.")
        
        self._is_connected_to_api = api_success

    def get_latest_frame(self):
        if self.camera:
            try:
                ret, frame = self.camera.read()
                if ret:
                    return frame
            except:
                pass
        return None

    def update_head_pose(self):
        if not self.base_url:
            return
        try:
            resp = requests.get(f"{self.base_url}/api/state/full", params={"with_head_pose": "true", "with_body_yaw": "true"}, timeout=1)
            if resp.status_code == 200:
                data = resp.json()
                pose = data.get("head_pose", {})
                if pose:
                    self.current_pitch = pose.get("pitch", 0.0)
                    self.current_roll = pose.get("roll", 0.0)
                    self.current_yaw = pose.get("yaw", 0.0)
                
                self.current_body_yaw = data.get("body_yaw", 0.0)
                if self.current_body_yaw is None: 
                    self.current_body_yaw = 0.0
        except:
            pass

    def move_head(self, d_yaw, d_pitch):
        """
        Non-blocking wrapper for move_head.
        """
        if self._is_moving:
            return
        
        self._is_moving = True
        threading.Thread(target=self._move_head_sync, args=(d_yaw, d_pitch)).start()

    def _move_head_sync(self, d_yaw, d_pitch):
        """
        Move head relative to current position.
        Also moves body if head turns too far.
        """
        if not self.base_url:
            self._is_moving = False
            return

        try:
            # Update current first
            self.update_head_pose()
            
            target_head_yaw = self.current_yaw + d_yaw
            target_pitch = self.current_pitch + d_pitch
            target_body_yaw = self.current_body_yaw
    
            # Body Follows Head Logic (Proportional Washout)
            # Smoothly transfer head rotation to body rotation to keep head centered
            # This allows the robot to "lean into" the turn continuously
            WASHOUT_GAIN = 0.5 
            
            transfer_yaw = target_head_yaw * WASHOUT_GAIN
            target_body_yaw += transfer_yaw
            target_head_yaw -= transfer_yaw
            
            # Clamp limits
            target_head_yaw = np.clip(target_head_yaw, -1.0, 1.0)     # Head limits
            target_pitch = np.clip(target_pitch, -0.8, 0.2) # Pitch limits [-0.8 (Up), 0.2 (Down)]
            
            # We don't clip body yaw strictly, but let's assume +/- 3.0 rad (~170 deg) is safe 
            target_body_yaw = np.clip(target_body_yaw, -3.0, 3.0)
    
            payload = {
                "head_pose": {
                    "roll": self.current_roll,
                    "pitch": target_pitch,
                    "yaw": target_head_yaw,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0 
                },
                "body_yaw": target_body_yaw,
                "duration": 0.2, # Fast update
                "interpolation": "linear"
            }
            
            requests.post(f"{self.base_url}/api/move/goto", json=payload, timeout=0.5)
        except:
            pass
        finally:
            self._is_moving = False

    def recenter_head(self):
        if self._is_moving:
            return
        self._is_moving = True
        threading.Thread(target=self._recenter_head_sync).start()

    def _recenter_head_sync(self):
        """Moves head to default center position (0, 0, 0)."""
        if not self.base_url:
            self._is_moving = False
            return
        
        payload = {
            "head_pose": {
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "x": 0.0,
                "y": 0.0,
                "z": 0.0
            },
            "body_yaw": 0.0,
            "duration": 1.0, # Smooth return
            "interpolation": "minjerk"
        }
        try:
            requests.post(f"{self.base_url}/api/move/goto", json=payload, timeout=1)
        except:
            pass
        finally:
            self._is_moving = False

    def wiggle_antennas(self):
        if not self.base_url:
            return
        
        def move_antennas(l_deg, r_deg, duration):
            l_rad = np.deg2rad(l_deg)
            r_rad = np.deg2rad(r_deg)
            payload = {
                "antennas": [l_rad, r_rad],
                "duration": duration,
                "interpolation": "minjerk"
            }
            try:
                requests.post(f"{self.base_url}/api/move/goto", json=payload, timeout=1)
            except:
                pass

        try:
            for _ in range(3):
                move_antennas(30, -30, 0.2)
                time.sleep(0.2)
                move_antennas(-10, 10, 0.2)
                time.sleep(0.2)
            move_antennas(0, 0, 0.5)
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
