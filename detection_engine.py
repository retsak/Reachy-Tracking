
import cv2
import numpy as np
import os

class DetectionEngine:
    def __init__(self):
        # Prevent OpenCV from spawning multiple threads which can cause contention on MacOS
        cv2.setNumThreads(0)
        
        # Paths to model
        self.prototxt = 'MobileNetSSD_deploy.prototxt'
        self.model = 'MobileNetSSD_deploy.caffemodel'
        
        # Pascal VOC classes
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.CAT_IDX = self.CLASSES.index("cat")
        self.PERSON_IDX = self.CLASSES.index("person")
        self.TARGET_IDXS = {self.CAT_IDX, self.PERSON_IDX}
        
        # Load Haar Cascade for Face Detection
        self.face_cascade = None
        try:
            # Hardcoding path found by `find` command earlier for robustness
            # venv/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml
            # But let's verify if cv2.data.haarcascades works in main.
            # safe fallback: using the standard path relative to site-packages if available
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(path):
                # Fallback to local copy if we ever downloaded it, or try to find it
                pass 
            self.face_cascade = cv2.CascadeClassifier(path)
            if self.face_cascade.empty():
                print("Error: Failed to load Haar Cascade xml")
            else:
                print("Face Detector (Haar) loaded successfully.")
        except Exception as e:
            print(f"Error loading face detector: {e}")

        self.net = None
        if os.path.exists(self.prototxt) and os.path.exists(self.model):
            try:
                self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    # Run a dummy forward pass to check if CUDA works
                    dummy_blob = np.random.standard_normal([1, 3, 300, 300]).astype(np.float32)
                    self.net.setInput(dummy_blob)
                    self.net.forward()
                    print("CUDA (GPU) initialized successfully.")
                except Exception as e:
                    print(f"CUDA initialization failed ({e}), falling back to CPU.")
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                print("MobileNet SSD loaded successfully.")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print(f"Model files not found: {self.prototxt}, {self.model}")

    def detect(self, frame, min_confidence: float = 0.4):
        """Detect cats and humans (person) in the frame using MobileNet SSD.

        Returns:
            (detected, best_box, best_label, best_confidence)
            - detected: bool
            - best_box: (x, y, w, h) in pixel coords or None
            - best_label: str or None ("cat" | "person")
            - best_confidence: float
        """
        if frame is None or self.net is None:
            return False, None, None, 0.0

        (h, w) = frame.shape[:2]
        # Resize to 300x300 and create blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        detected = False
        best_box = None
        best_label = None
        highest_confidence = 0.0

        # 1. Run Haar Cascade (Face) - Priority!
        if self.face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # If faces found, pick the largest one
            if len(faces) > 0:
                # print(f"DEBUG: Found {len(faces)} faces", flush=True)
                # Face is usually the best target.
                # Find largest face
                largest_area = 0
                for (fx, fy, fw, fh) in faces:
                    area = fw * fh
                    if area > largest_area:
                        largest_area = area
                        best_box = (int(fx), int(fy), int(fw), int(fh))
                        best_label = "face"
                        highest_confidence = 1.0 # Haar doesn't provide conf, assume high
                        detected = True

        # 2. Run MobileNet SSD (Person/Cat) - Only if no face or allow override?
        # Let's run it to see if we find a cat (which might be higher priority than person-body)
        # But if we found a face, we probably want to prioritize it over "person" body.
        
        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            # Filter weak detections and ensure it's a target we care about
            # Filter weak detections and ensure it's a target we care about
            if confidence > 0.4 and idx in self.TARGET_IDXS:
                label = self.CLASSES[idx]
                
                # Priority Logic: FACE > CAT > PERSON
                # If we already have a face (confidence=1.0), only override if we find a CAT (maybe?) 
                # Actually user wants Faces. So Face > All?
                # Let's say Face > Person. Cat vs Face? "Look for faces and not just the person".
                # Implies Face > Person. 
                # Let's treat Face as max priority (1.0 conf).
                # MobileNet confidence is usually 0.4-0.9.
                # So if best_label is "face", we likely stick with it unless this detection is ...?
                
                # Let's just store candidates and pick best at end?
                # No, standard loop replacement.
                
                is_better = False
                if not detected:
                    is_better = True
                else:
                    # Current best is present.
                    if best_label == "face":
                        # Face wins over person/cat usually for eye contact
                        is_better = False 
                    elif label == "cat" and best_label == "person":
                        is_better = True # Cat > Person
                    elif confidence > highest_confidence and label == best_label:
                        is_better = True # Better version of same class
                
                if is_better:
                    detected = True
                    
                    # Compute box if we haven't already
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Clamp coordinates
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)
                    
                    width = endX - startX
                    height = endY - startY
                    best_box = (startX, startY, width, height)
                    best_label = label
                    highest_confidence = confidence

        return detected, best_box, best_label, highest_confidence
