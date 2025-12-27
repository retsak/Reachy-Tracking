import cv2
import time
import platform

def test_camera_properties():
    print(f"OS: {platform.system()}")
    print("Initializing camera...")
    
    # Try different backends
    backends = [cv2.CAP_ANY]
    if platform.system() == 'Darwin':
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    
    cap = None
    for backend in backends:
        print(f"Trying backend: {backend}")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print("Camera opened successfully.")
            break
        
    if not cap or not cap.isOpened():
        print("Failed to open camera.")
        return

    # Properties to test
    props = {
        "Brightness": cv2.CAP_PROP_BRIGHTNESS,
        "Contrast": cv2.CAP_PROP_CONTRAST,
        "Saturation": cv2.CAP_PROP_SATURATION,
        "Gain": cv2.CAP_PROP_GAIN,
        "Exposure": cv2.CAP_PROP_EXPOSURE,
        "Auto Exposure": cv2.CAP_PROP_AUTO_EXPOSURE
    }
    
    print("\n--- Initial Values ---")
    for name, prop_id in props.items():
        val = cap.get(prop_id)
        print(f"{name}: {val}")
        
    print("\n--- Modification Test ---")
    # Try to set Exposure (often fails on Mac)
    print("Attempting to set Exposure to -5...")
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    time.sleep(1)
    print(f"New Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

    print("Attempting to set Gain to 100...")
    cap.set(cv2.CAP_PROP_GAIN, 100)
    time.sleep(1)
    print(f"New Gain: {cap.get(cv2.CAP_PROP_GAIN)}")
    
    cap.release()
    print("\nDiagnostics complete.")

if __name__ == "__main__":
    test_camera_properties()
