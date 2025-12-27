import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"

def log(msg):
    print(f"[TEST] {msg}")

def check_status():
    try:
        resp = requests.get(f"{BASE_URL}/api/daemon/status", timeout=2)
        log(f"Status: {resp.status_code}")
        log(f"Payload: {json.dumps(resp.json(), indent=2)}")
    except Exception as e:
        log(f"Status Check Failed: {e}")

def get_joints():
    # Try to find what joints exist
    endpoints = [
        "/api/joints",
        "/api/move/joints",
        "/api/forward_kinematics/joints"
    ]
    for ep in endpoints:
        try:
            resp = requests.get(f"{BASE_URL}{ep}", timeout=2)
            if resp.status_code == 200:
                log(f"Found joints at {ep}: {resp.json().keys()}")
                return resp.json()
        except:
            pass
    log("Could not list joints.")
    return {}

def test_antennas():
    log("--- Testing Antennas ---")
    # Try 1: Joints payload (current impl)
    payload = {
        "joints": {
            "l_antenna_pitch": 0.5,
            "r_antenna_pitch": -0.5
        },
        "duration": 1.0,
        "interpolation": "minjerk"
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/move/goto", json=payload, timeout=2)
        log(f"Antenna 'joints' attempt: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"Antenna 'joints' exc: {e}")

    # Try 2: High level 'antennas' list (old impl)
    payload_v2 = {
        "antennas": [0.5, -0.5],
        "duration": 1.0
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/move/goto", json=payload_v2, timeout=2)
        log(f"Antenna 'antennas' list attempt: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"Antenna 'antennas' list exc: {e}")

def test_base():
    log("--- Testing Base ---")
    # Try 1: body_yaw in move/goto
    payload = {
        "body_yaw": 25.0, # degrees? radians? Usually radians in API but let's see. 25 rad is huge. 0.5 rad (~30 deg).
        "duration": 1.0
    }
    # Reachy API usually expects degrees for some high level, radians for low level.
    # Let's try 0.5 (radians)
    payload["body_yaw"] = 0.5
    try:
        resp = requests.post(f"{BASE_URL}/api/move/goto", json=payload, timeout=2)
        log(f"Base 'body_yaw' attempt: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"Base 'body_yaw' exc: {e}")

def test_head_smoothness():
    log("--- Testing Head Smoothness ---")
    # Move head to one side slowly
    payload = {
        "head_pose": {
            "roll": 0, "pitch": 0, "yaw": 20, # degrees?
            "x": 0, "y": 0, "z": 0
        },
        "duration": 1.0,
        "interpolation": "minjerk"
    }
    # Reachy 'head_pose' usually takes DEGREES for roll/pitch/yaw if using high level? 
    # Or Radians? Standard is usually Radians for everything internal.
    # 20 rad is spinning like crazy. 0.3 rad is ~20 deg.
    payload["head_pose"]["yaw"] = 0.3
    
    try:
        resp = requests.post(f"{BASE_URL}/api/move/goto", json=payload, timeout=2)
        log(f"Head Smooth Move: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"Head Smooth Move exc: {e}")

def check_compliance():
    log("--- Checking Compliance ---")
    # Usually /api/joints or similar lists compliance
    # Try endpoints
    try:
        resp = requests.get(f"{BASE_URL}/api/joints", timeout=2)
        if resp.status_code == 200:
            joints = resp.json()
            # print(json.dumps(joints, indent=2)) 
            # Check a few key joints
            for j in ["l_antenna_pitch", "head_pitch", "r_antenna_pitch"]:
                if j in joints:
                    log(f"Joint {j}: {joints[j]}")
        else:
            log(f"Failed to get joints: {resp.status_code}")
    except Exception as e:
        log(f"Compliance check exc: {e}")

def set_stiff():
    log("--- Setting Stiffness (Turning ON) ---")
    # Turn on all joints
    # Reachy Mini API might have a global turn_on?
    # Or set compliance to false for all.
    payload = {
        "compliant": False
    }
    # Try generic turn_on if exists, otherwise assume compliance endpoint
    # Common endpoint: /api/joints/{name}/compliance
    # Or bulk in /api/joints_compliance
    
    # Try bulk update if possible, or iterating
    # First, let's try getting joint names again to iterate
    pass 

def test_garbage_joint():
    log("--- Testing Garbage Joint Name ---")
    payload = {
        "joints": {
            "fake_joint_999": 1.0
        },
        "duration": 1.0
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/move/goto", json=payload, timeout=2)
        log(f"Garbage Joint: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"Garbage Joint exc: {e}")

def probe_turn_on():
    log("--- Probing Turn On ---")
    eps = ["/api/turn_on", "/api/compliant/false", "/api/motors/on"]
    for ep in eps:
        try:
            resp = requests.post(f"{BASE_URL}{ep}", json={}, timeout=1)
            log(f"Probe {ep}: {resp.status_code}")
        except:
            pass

def test_mobile_base_key():
    log("--- Testing 'mobile_base' key ---")
    payload = {
        "mobile_base": {
            "theta": 0.5 # radians
        },
        "duration": 1.0
    }
    try:
        resp = requests.post(f"{BASE_URL}/api/move/goto", json=payload, timeout=2)
        log(f"Mobile Base Key: {resp.status_code} - {resp.text}")
    except Exception as e:
        log(f"Mobile Base Key exc: {e}")

if __name__ == "__main__":
    check_status()
    # check_compliance()
    test_garbage_joint()
    test_mobile_base_key()
    probe_turn_on()
    test_antennas() # Re-run to see if turn-on helped if found
