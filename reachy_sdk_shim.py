
import requests
import numpy as np

class ReachyMini:
    def __init__(self, host='localhost'):
        self.host = host
        # Handle cases where host might already have http
        if "://" in host:
             self.base_url = host
        else:
             self.base_url = f"http://{host}:8000"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def goto_target(self, head=None, antennas=None, body_yaw=None, duration=1.0, method="minjerk"):
        """
        Move the robot to a target configuration.
        
        Args:
            head (dict): Head pose dict (x, y, z, roll, pitch, yaw) or None
            antennas (list/array): [left_pitch, right_pitch] in radians or None
            body_yaw (float): Body yaw in radians or None
            duration (float): Time in seconds
            method (str): "minjerk" or "linear"
        """
        payload = {
            "duration": duration,
            "interpolation": method
        }
        
        if head is not None:
            # Flatten helper object if needed, though dict is expected
            payload["head_pose"] = head
            
        if antennas is not None:
            # Ensure it's a list
            if hasattr(antennas, "tolist"):
                antennas = antennas.tolist()
            payload["antennas"] = antennas
            
        if body_yaw is not None:
            payload["body_yaw"] = float(body_yaw)
            # Shotgun support
            payload["mobile_base"] = {"theta": float(body_yaw)}

        try:
            requests.post(f"{self.base_url}/api/move/goto", json=payload, timeout=0.5)
        except Exception as e:
            # print(f"SDK Shim Error: {e}")
            pass

def create_head_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, mm=False):
    """
    Helper to create head pose dictionary.
    Supports 'mm=True' if user passes millimeters (API expects meters usually).
    """
    # API usually expects meters for x,y,z and radians for r,p,y
    # If mm is True, convert to meters
    scale = 0.001 if mm else 1.0
    
    return {
        "x": x * scale,
        "y": y * scale,
        "z": z * scale,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw
    }
