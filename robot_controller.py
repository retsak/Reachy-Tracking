import numpy as np
import time
import requests
import cv2
import threading
import platform
import queue
import soundfile as sf
import scipy.signal
import os
import tempfile
from reachy_mini import ReachyMini as SDKReachyMini
from reachy_sdk_shim import ReachyMini, create_head_pose

class RobotController:
    def __init__(self, host='localhost'):
        self.host = host
        self.base_url = None
        self.camera = None
        self._is_connected_to_api = False
        # Windows DirectShow exposure defaults (tune as needed)
        self.win_exposure_mode = 0.75   # 0.75=manual, 0.25=auto
        self.win_exposure_value = -4    # less negative -> brighter on many webcams
        self.win_gain = 12.0            # moderate gain boost for dim scenes
        
        # Command Queue for serialized execution
        self.command_queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self._worker_thread.start()
        
        # State Monitor Worker
        self._state_thread = threading.Thread(target=self._state_worker, daemon=True)
        self._state_thread.start()
        
        # Camera State
        self._latest_camera_frame = None
        self._camera_lock = threading.Lock()
        self._camera_thread = None
        self._reapply_camera_settings = False
        self._pause_camera_reads = False
        
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.current_roll = 0.0
        self.current_body_yaw = 0.0
        self.current_antenna_left = 0.0
        self.current_antenna_right = 0.0
        
        # Motion Overlays
        self._speech_roll_offset = 0.0
        self.current_base_roll = 0.0

        # Audio volume (0.0 - 1.0). Default 25%.
        self.audio_volume = 0.5

    def set_audio_volume(self, volume: float):
        """Set speaker volume (0.0 - 1.0)."""
        try:
            v = float(volume)
        except Exception:
            return
        v = max(0.0, min(1.0, v))
        self.audio_volume = v
        print(f"[AUDIO] Volume set to {int(v*100)}%")

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
                    elif platform.system() == 'Darwin':
                         # Force AVFoundation on Mac
                        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                    else:
                        cap = cv2.VideoCapture(idx) 
                        
                    if not cap.isOpened():
                         cap = cv2.VideoCapture(idx) # Fallback
                    
                    if cap.isOpened():
                        # Force hardware settings BEFORE reading a frame to update driver mode
                        self._enforce_camera_config(cap)

                        ret, _ = cap.read()
                        if ret:
                            self.camera = cap
                            print(f"Camera initialized on index {idx} (MJPEG 15FPS Requested).")

                            try:
                                fourcc = int(self.camera.get(cv2.CAP_PROP_FOURCC))
                                fps = self.camera.get(cv2.CAP_PROP_FPS)
                                width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                                height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                auto_exp = self.camera.get(cv2.CAP_PROP_AUTO_EXPOSURE)
                                exp_val = self.camera.get(cv2.CAP_PROP_EXPOSURE)
                                gain_val = self.camera.get(cv2.CAP_PROP_GAIN)
                                # Log the actual mode the driver applied for quick debugging
                                print(
                                    f"[CAMERA] Mode -> {width:.0f}x{height:.0f} @ {fps:.1f} FPS, "
                                    f"FOURCC={fourcc:08x}, AutoExposure={auto_exp:.2f}, "
                                    f"Exposure={exp_val:.2f}, Gain={gain_val:.1f}"
                                )
                            except Exception as elog:
                                print(f"[CAMERA] Failed to read back mode: {elog}")
                            
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

    def _enforce_camera_config(self, cap):
        """Force MJPEG + 640x480 @15fps and enable auto exposure where supported."""
        try:
            # Set MJPEG first to allow higher resolutions/rates without USB bandwidth choke
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Platform-specific exposure settings
            if platform.system() == 'Windows':
                # Windows DirectShow: 0.75=manual, 0.25=auto
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.win_exposure_mode)
                cap.set(cv2.CAP_PROP_EXPOSURE, self.win_exposure_value)
                cap.set(cv2.CAP_PROP_GAIN, self.win_gain)
            elif platform.system() == 'Darwin':
                # macOS AVFoundation: Lower FPS = longer exposure time = brighter image
                cap.set(cv2.CAP_PROP_FPS, 10)  # Even lower for max exposure
                
                # Try to enable auto-exposure (3.0 = auto mode on many cameras)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0)
                
                # Boost brightness if supported
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
                
                # Some cameras support these on Mac
                cap.set(cv2.CAP_PROP_GAIN, 15.0)
                
                print("[CAMERA] macOS: Forced 10 FPS + auto exposure for brightness")
            else:
                cap.set(cv2.CAP_PROP_FPS, 15)

            # Read back and, if the driver ignored settings, try once more at 640x480/15
            try:
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if w > 640 or h > 480 or fps > 20:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 10 if platform.system() == 'Darwin' else 15)
            except Exception:
                pass
        except Exception as eobj:
             print(f"Warning: Failed to set camera config: {eobj}")

    def _camera_worker(self):
        """Background thread to read frames as fast as possible (latencyless)."""
        while self.camera and self.camera.isOpened():
            try:
                # 0. Check if camera reads are paused (e.g., during audio playback)
                if self._pause_camera_reads:
                    time.sleep(0.05)
                    continue
                
                # 1. Check if we need to enforce settings (e.g. after audio playback reset)
                if self._reapply_camera_settings:
                    print("[CAMERA] Re-applying MJPEG/15FPS settings...")
                    self._enforce_camera_config(self.camera)
                    self._reapply_camera_settings = False
                
                ret, frame = self.camera.read()
                if ret:
                    # Fix color space issues (Mac often returns BGRA or YUYV)
                    if frame.ndim == 3:
                        if frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                        elif frame.shape[2] == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
                    
                    # SOFTWARE BRIGHTNESS BOOST for macOS (camera ignores hardware settings)
                    if platform.system() == 'Darwin':
                        # Boost brightness by 1.8x and add exposure compensation
                        frame = cv2.convertScaleAbs(frame, alpha=1.8, beta=40)
                    
                    with self._camera_lock:
                        self._latest_camera_frame = frame
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[CAMERA] Read error: {e}")
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

            # Constants for Safety Limits (converted to radians)
            MAX_HEAD_PITCH = np.deg2rad(40.0)
            MAX_HEAD_YAW = np.deg2rad(180.0)
            MAX_BODY_YAW = np.deg2rad(160.0)
            MAX_YAW_DELTA = np.deg2rad(65.0)

            # Clamp Body First (Absolute Limit)
            target_body_yaw = np.clip(target_body_yaw, -MAX_BODY_YAW, MAX_BODY_YAW)
            
            # Clamp Head Absolute Limits
            target_head_yaw = np.clip(target_head_yaw, -MAX_HEAD_YAW, MAX_HEAD_YAW)
            target_pitch = np.clip(target_pitch, -MAX_HEAD_PITCH, MAX_HEAD_PITCH)
            
            # Clamp Head Relative to Body (Mechanical Constraint)
            # Head Yaw must be within [BodyYaw - 65deg, BodyYaw + 65deg]
            min_head_limit = target_body_yaw - MAX_YAW_DELTA
            max_head_limit = target_body_yaw + MAX_YAW_DELTA
            target_head_yaw = np.clip(target_head_yaw, min_head_limit, max_head_limit)

            # USE NEW SDK SHIM
            with ReachyMini(self.host) as mini:
                # DEBUG LOGGING (User Requested)
                print(f"[DEBUG] Zone Logic: d_yaw_err={d_yaw:.3f} | Zone={'SIDE' if abs(d_yaw) > ZONE_THRESHOLD else 'CENTER'}", flush=True)
                print(f"[DEBUG] Target: HeadYaw={target_head_yaw:.3f} BodyYaw={target_body_yaw:.3f} Pitch={target_pitch:.3f}", flush=True)
                
                mini.goto_target(
                    head=create_head_pose(
                        roll=0.0,
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
                self.current_body_yaw = target_body_yaw
                self.current_pitch = target_pitch
                self.current_roll = 0.0 # This is base roll logic?
                self.current_base_roll = 0.0
                # self.current_yaw keeps track of head relative to body? 
                # In this logic current_yaw is treated as head position.
                self.current_yaw = target_head_yaw
        except:
            pass
            
    def set_pose(self, head_yaw=None, head_pitch=None, head_roll=None, body_yaw=None, antenna_left=None, antenna_right=None, duration=2.0):
        """Enqueue absolute pose command for Manual Control."""
        self.command_queue.put((self._set_pose_sync, (head_yaw, head_pitch, head_roll, body_yaw, antenna_left, antenna_right, duration)))
    
    def _set_pose_sync(self, h_yaw, h_pitch, h_roll, b_yaw, a_left, a_right, duration):
        """Execute absolute move."""
        # Use current state for missing values
        if h_yaw is None: h_yaw = self.current_yaw
        if h_pitch is None: h_pitch = self.current_pitch
        
        # Base Roll Management
        if h_roll is not None:
            # explicit update (from tracking or manual) -> update base
            self.current_base_roll = h_roll
        else:
            # implicit update -> use existing base
            h_roll = self.current_base_roll
            
        if b_yaw is None: b_yaw = self.current_body_yaw
        if a_left is None: a_left = self.current_antenna_left
        if a_right is None: a_right = self.current_antenna_right

        # Apply Speech Offset to Base
        # We store the *Physical* roll in current_roll for reference, but use Base for logic
        physical_roll = h_roll + self._speech_roll_offset
        # Update internal "Physical" state to match what we send
        self.current_roll = physical_roll
        
        # Override h_roll for sending
        h_roll = physical_roll
        if not self.base_url:
            return 
            
        try:
             # Update internal state
            self.current_roll = h_roll
            self.current_body_yaw = b_yaw
            self.current_antenna_left = a_left
            self.current_antenna_right = a_right
            
            # --- SAFETY CLAMPS ---
            MAX_HEAD_PITCH = np.deg2rad(40.0)
            MAX_HEAD_ROLL = np.deg2rad(40.0)
            MAX_HEAD_YAW = np.deg2rad(180.0)
            MAX_BODY_YAW = np.deg2rad(160.0)
            MAX_YAW_DELTA = np.deg2rad(65.0)
            
            # Clamp Absolute
            b_yaw = np.clip(b_yaw, -MAX_BODY_YAW, MAX_BODY_YAW)
            h_yaw = np.clip(h_yaw, -MAX_HEAD_YAW, MAX_HEAD_YAW)
            h_pitch = np.clip(h_pitch, -MAX_HEAD_PITCH, MAX_HEAD_PITCH)
            h_roll = np.clip(h_roll, -MAX_HEAD_ROLL, MAX_HEAD_ROLL)
            
            # Clamp Relative (Head vs Body)
            min_head = b_yaw - MAX_YAW_DELTA
            max_head = b_yaw + MAX_YAW_DELTA
            h_yaw = np.clip(h_yaw, min_head, max_head)
            # ---------------------
             
            with ReachyMini(self.host) as mini:
                mini.goto_target(
                    head=create_head_pose(
                        roll=h_roll,
                        pitch=h_pitch,
                        yaw=h_yaw,
                        x=0, y=0, z=0
                    ),
                    body_yaw=b_yaw,
                    antennas=np.deg2rad([a_left, a_right]),
                    duration=duration,
                    method="minjerk"
                )
        except Exception as e:
            print(f"Set Pose Error: {e}")

    def _state_worker(self):
        """Periodically fetch robot state to keep UI in sync."""
        while True:
            if self._is_connected_to_api:
                self.update_head_pose()
            time.sleep(0.1)

    def update_head_pose(self):
        """Fetch current head pose from API using /api/state/full."""
        if not self.base_url: return
        try:
             with ReachyMini(self.host) as mini:
                 state = mini.get_joints()
                 if not state: return
                 
                 # Structure: {'head_pose': {'x':..., 'roll':...}, 'body_yaw': float, 'antennas_position': [l, r]}
                 
                 # Head (Radians)
                 if "head_pose" in state and state["head_pose"]:
                     hp = state["head_pose"]
                     self.current_pitch = float(hp.get("pitch", self.current_pitch))
                     self.current_roll = float(hp.get("roll", self.current_roll))
                     self.current_yaw = float(hp.get("yaw", self.current_yaw))
                 
                 # Body (Radians)
                 if "body_yaw" in state and state["body_yaw"] is not None:
                     self.current_body_yaw = float(state["body_yaw"])

                 # Antennas (Convert Rad -> Deg)
                 if "antennas_position" in state and state["antennas_position"]:
                     ants = state["antennas_position"]
                     if len(ants) >= 2:
                         self.current_antenna_left = np.rad2deg(float(ants[0]))
                         self.current_antenna_right = np.rad2deg(float(ants[1]))
                 
        except Exception:
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

    def set_motor_mode(self, mode):
        """
        mode: 'stiff', 'limp', 'soft'
        """
        self.command_queue.put((self._set_motor_mode_sync, (mode,)))

    def _set_motor_mode_sync(self, mode):
        if not self.base_url: return
        try:
            with ReachyMini(self.host) as mini:
                if mode == 'stiff':
                    mini.enable_motors()
                elif mode == 'limp':
                    mini.disable_motors()
                elif mode == 'soft':
                    mini.enable_gravity_compensation()
        except Exception as e:
            print(f"Motor Mode Error: {e}")

    def play_sound(self, filename: str):
        """Play a wav file by pushing samples to the audio device via SDK."""
        # Use a background thread to avoid blocking
        threading.Thread(target=self._play_sound_sync, args=(filename,), daemon=True).start()

    def play_sound_from_file(self, filepath: str):
        """Play audio from an absolute file path (used for TTS output)."""
        threading.Thread(target=self._play_audio_file, args=(filepath,), daemon=True).start()

    def _play_audio_file(self, filepath: str):
        """Internal method to play audio from any file path."""
        if not os.path.exists(filepath):
            print(f"[AUDIO] File not found: {filepath}")
            return

        print(f"[AUDIO] Playing from: {filepath}")
        tempdir = os.path.abspath(tempfile.gettempdir())
        is_temp_file = os.path.abspath(filepath).startswith(tempdir)
        
        # Mark camera for settings reapplication before and after audio
        self._reapply_camera_settings = True

        # Try multiple times with increasing timeouts
        for attempt in range(3):
            try:
                timeout = 10 + (attempt * 5)  # 10s, 15s, 20s
                print(f"[AUDIO] Attempt {attempt + 1}: connecting with {timeout}s timeout...")
                
                with SDKReachyMini(log_level="ERROR", media_backend="default", timeout=timeout) as mini:
                    data, samplerate_in = sf.read(filepath, dtype="float32")

                    output_rate = mini.media.get_output_audio_samplerate()
                    if samplerate_in != output_rate:
                        new_len = int(len(data) * (output_rate / samplerate_in))
                        data = scipy.signal.resample(data, new_len)

                    if data.ndim > 1:
                        data = np.mean(data, axis=1)

                    # Apply volume scaling
                    try:
                        data = (data * float(self.audio_volume)).astype('float32')
                    except Exception:
                        pass

                    # Start Speaking Motion
                    self._is_speaking = True
                    self._speech_roll_offset = 0.0

                    def motion_thread():
                        # Simple alternating sway for gentle, continuous motion
                        direction = 1
                        while self._is_speaking:
                            # +/- 0.375 degrees offset
                            target_offset = 0.375 * direction
                            
                            # Set offset immediately (no manual interpolation)
                            self._speech_roll_offset = target_offset
                            
                            # Push update with LONG duration for smoothness
                            try:
                                self.set_pose(duration=1.5) 
                            except: pass
                            
                            # Wait for motion
                            for _ in range(15): # 1.5s
                                if not self._is_speaking: break
                                time.sleep(0.1)
                            
                            direction *= -1

                        # Reset gently
                        self._speech_roll_offset = 0.0
                        try: self.set_pose(duration=2.0)
                        except: pass

                    threading.Thread(target=motion_thread, daemon=True).start()

                    mini.media.start_playing()
                    print(f"[AUDIO] Playing {len(data)} samples...")
                    # Push samples in chunks
                    chunk_size = 1024
                    for i in range(0, len(data), chunk_size):
                        chunk = data[i : i + chunk_size]
                        mini.media.push_audio_sample(chunk)

                    # Wait for playback to complete
                    time.sleep(len(data) / float(output_rate) + 0.5)
                    self._is_speaking = False # Stop motion
                    mini.media.stop_playing()
                    print(f"[AUDIO] Finished: {filepath}")
                    
                    # Trigger camera settings reapplication after audio
                    self._reapply_camera_settings = True
                    
                    # Delete temporary file if it was created by TTS
                    if is_temp_file:
                        try:
                            os.remove(filepath)
                            print(f"[AUDIO] Deleted temp file: {filepath}")
                        except Exception as de:
                            print(f"[AUDIO] Temp delete failed: {de}")
                    return  # Success
            except TimeoutError:
                if attempt < 2:
                    print(f"[AUDIO] Timeout, retrying with longer timeout...")
                    time.sleep(1)
                else:
                    print(f"[AUDIO] Failed to connect after 3 attempts")
                    # Cleanup on failure for temp files
                    if is_temp_file:
                        try:
                            os.remove(filepath)
                            print(f"[AUDIO] Deleted temp file after failure: {filepath}")
                        except Exception as de:
                            print(f"[AUDIO] Temp delete failed: {de}")
            except Exception as e:
                print(f"[AUDIO] Error playing file: {e}")
                # Cleanup on error for temp files
                if is_temp_file:
                    try:
                        os.remove(filepath)
                        print(f"[AUDIO] Deleted temp file after error: {filepath}")
                    except Exception as de:
                        print(f"[AUDIO] Temp delete failed: {de}")

    # --- EMOTIVE MOTIONS ---
    def trigger_emotion(self, emotion_name: str):
        """Trigger a predefined emotion trajectory in a non-blocking thread."""
        def run_emotion():
            print(f"[EMOTION] Playing: {emotion_name}")
            
            # Helper to reset to neutral
            def reset(dur=1.0):
                self.set_pose(
                    head_roll=0.0, 
                    head_pitch=0.0, 
                    head_yaw=0.0, 
                    antenna_left=0.0, 
                    antenna_right=0.0, 
                    duration=dur
                )

            if emotion_name == "happy":
                # Quick antenna wiggles + head tilt
                self.set_pose(head_roll=0.3, head_pitch=-0.2, antenna_left=30, antenna_right=-30, duration=0.4)
                time.sleep(0.4)
                self.set_pose(head_roll=-0.3, antenna_left=-30, antenna_right=30, duration=0.4)
                time.sleep(0.4)
                self.set_pose(head_roll=0.3, antenna_left=30, antenna_right=-30, duration=0.4)
                time.sleep(0.4)
                self.set_pose(head_roll=0.0, antenna_left=0.0, antenna_right=0.0, duration=0.5)

            elif emotion_name == "sad":
                # Head down, antennas droop
                self.set_pose(head_pitch=30.0, antenna_left=50.0, antenna_right=50.0, duration=1.5)
                time.sleep(2.0)
                reset(2.0)

            elif emotion_name == "no":
                # Shake head (Yaw)
                mag = 10.0 # degrees
                speed = 0.4
                self.set_pose(head_yaw=mag, duration=speed)
                time.sleep(speed)
                self.set_pose(head_yaw=-mag, duration=speed)
                time.sleep(speed)
                self.set_pose(head_yaw=mag, duration=speed)
                time.sleep(speed)
                self.set_pose(head_yaw=0.0, duration=0.5)

            elif emotion_name == "yes":
                # Nod head (Pitch)
                mag = 7.5 # degrees
                speed = 0.4
                self.set_pose(head_pitch=mag, duration=speed)
                time.sleep(speed)
                self.set_pose(head_pitch=-2.5, duration=speed)
                time.sleep(speed)
                self.set_pose(head_pitch=mag, duration=speed)
                time.sleep(speed)
                self.set_pose(head_pitch=0.0, duration=0.5)

            elif emotion_name == "confused":
                # Head tilt + one antenna up
                self.set_pose(head_roll=0.4, head_pitch=-0.1, antenna_left=40.0, antenna_right=-10.0, duration=1.0)
                time.sleep(2.0)
                reset(1.0)

        threading.Thread(target=run_emotion, daemon=True).start()

    def _play_sound_sync(self, filename: str):
        if not self.host:
            print("Cannot play sound: No host known.")
            return

        # Resolve path (default voice directory)
        base_path = os.path.join(os.getcwd(), "Audio", "Default Voice")
        file_path = os.path.join(base_path, filename)

        if not os.path.exists(file_path):
            print(f"[AUDIO] File not found: {file_path}")
            return

        print(f"[AUDIO] Playing: {filename}")
        
        # Mark camera for settings reapplication before and after audio
        self._reapply_camera_settings = True

        # Try multiple times with increasing timeouts
        for attempt in range(3):
            try:
                timeout = 10 + (attempt * 5)  # 10s, 15s, 20s
                print(f"[AUDIO] Attempt {attempt + 1}: connecting with {timeout}s timeout...")
                
                with SDKReachyMini(log_level="ERROR", media_backend="default", timeout=timeout) as mini:
                    data, samplerate_in = sf.read(file_path, dtype="float32")

                    # Resample if needed
                    output_rate = mini.media.get_output_audio_samplerate()
                    if samplerate_in != output_rate:
                        new_len = int(len(data) * (output_rate / samplerate_in))
                        data = scipy.signal.resample(data, new_len)
                
                    # Mono conversion
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)

                    # Apply volume scaling
                    try:
                        data = (data * float(self.audio_volume)).astype('float32')
                    except Exception:
                        pass

                    mini.media.start_playing()
                    print(f"[AUDIO] Playing {len(data)} samples...")
                    # Push samples in chunks
                    chunk_size = 1024
                    for i in range(0, len(data), chunk_size):
                        chunk = data[i : i + chunk_size]
                        mini.media.push_audio_sample(chunk)

                    # Wait for playback to complete
                    time.sleep(len(data) / float(output_rate) + 0.5)
                    mini.media.stop_playing()
                    print(f"[AUDIO] Finished: {filename}")
                    
                    # Trigger camera settings reapplication after audio
                    self._reapply_camera_settings = True
                    
                    return  # Success
            except TimeoutError:
                if attempt < 2:
                    print(f"[AUDIO] Timeout, retrying with longer timeout...")
                    time.sleep(1)
                else:
                    print(f"[AUDIO] Failed to connect after 3 attempts")
            except Exception as e:
                print(f"[AUDIO] Error playing sound: {e}")
