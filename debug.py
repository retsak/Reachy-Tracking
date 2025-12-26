
print("Importing cv2...")
import cv2
print("Importing requests...")
import requests
print("Importing RobotController...")
from robot_controller import RobotController
print("Importing DetectionEngine...")
from detection_engine import DetectionEngine

print("Initializing DetectionEngine...")
de = DetectionEngine()
print(f"DetectionEngine initialized. Path: {de.cascade_path}")

print("Initializing RobotController...")
rc = RobotController()
print("Connecting...")
rc.connect()
print(f"RobotController connected: {rc.is_connected}")
if rc.is_connected:
    print("Testing wiggle...")
    rc.wiggle_antennas()

print("Done.")
