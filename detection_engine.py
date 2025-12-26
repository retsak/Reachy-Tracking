
import cv2
import numpy as np
import os

class DetectionEngine:
    def __init__(self):
        # Paths to model
        self.prototxt = 'MobileNetSSD_deploy.prototxt'
        self.model = 'MobileNetSSD_deploy.caffemodel'
        
        # Pascal VOC classes
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]
        self.CAT_IDX = self.CLASSES.index("cat")
        
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

    def detect(self, frame):
        """
        Detects cats in the frame using MobileNet SSD.
        Returns (detected: bool, best_box: (x, y, w, h) or None, annotated_frame)
        """
        if frame is None or self.net is None:
            return False, None, frame

        (h, w) = frame.shape[:2]
        # Resize to 300x300 and create blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        annotated_frame = frame.copy()
        detected = False
        best_box = None
        highest_confidence = 0.0

        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            # Filter weak detections and ensure it's a cat
            if confidence > 0.4 and idx == self.CAT_IDX:
                detected = True
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Clamp coordinates
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Draw prediction
                label = f"CAT: {confidence*100:.1f}%"
                cv2.rectangle(annotated_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y_label = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(annotated_frame, label, (startX, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Keep track of most confident detection
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    width = endX - startX
                    height = endY - startY
                    best_box = (startX, startY, width, height)

        return detected, best_box, annotated_frame
