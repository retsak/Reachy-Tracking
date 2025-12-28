
import cv2
import numpy as np
import os

class DetectionEngine:
    def __init__(self):
        # Prevent OpenCV from spawning multiple threads which can cause contention on MacOS
        cv2.setNumThreads(0)
        
        # Paths to model
        self.model_path = 'models/yolov8n.onnx'
        
        # COCO Classes (YOLOv8 default)
        self.CLASSES = {
            0: "person",
            15: "cat"
        }
        
        # Load Haar Cascade for Face Detection (Keep as priority/augment)
        self.face_cascade = None
        try:
            # Try site-packages path first if known, or standard location
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                 path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            else:
                 path = "haarcascade_frontalface_default.xml" # Local fallback

            if not os.path.exists(path):
                # Fallback to local copy if we ever downloaded it
                pass 
            
            self.face_cascade = cv2.CascadeClassifier(path)
            if self.face_cascade.empty():
                print("Error: Failed to load Haar Cascade xml")
            else:
                print("Face Detector (Haar) loaded successfully.")
        except Exception as e:
            print(f"Error loading face detector: {e}")

        self.net = None
        if os.path.exists(self.model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                
                # Check for CUDA/Metal? OpenCV DNN on Mac usually CPU or OpenCL
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                except:
                    pass
                
                print("YOLOv8 (ONNX) loaded successfully.")
            except Exception as e:
                print(f"Failed to load YOLOv8 model: {e}")
        else:
            print(f"Model file not found: {self.model_path}")

    def detect(self, frame, min_confidence: float = 0.5):
        """
        Detect faces, persons, and cats.
        
        Returns:
            list of dicts: [{'box': (x,y,w,h), 'label': str, 'conf': float}, ...]
        """
        results = []
        
        if frame is None:
            return results

        (h, w) = frame.shape[:2]

        # 1. Haar Face Detection (Priority)
        if self.face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30)
            )
            for (fx, fy, fw, fh) in faces:
                results.append({
                    "box": (int(fx), int(fy), int(fw), int(fh)),
                    "label": "face",
                    "conf": 0.5 # Haar is noisy, treat as low confidence unless validated
                })

        # 2. YOLOv8 Detection
        if self.net:
            # Preprocess
            # YOLOv8 expects 640x640, normalized 0-1
            # swapRB=True (OpenCV is BGR, Model trained on RGB usually, YOLOv8 handles this but usually swapRB=True)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Forward pass
            # Output shape: (1, 84, 8400) -> 4 box + 80 classes
            outputs = self.net.forward()
            
            # Post-Process
            # Transpose to (8400, 84) to make rows = detections
            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]
            
            boxes = []
            scores = []
            class_ids = []
            
            # Extract scaling factors
            x_factor = w / 640
            y_factor = h / 640
            
            data = outputs[0] # (8400, 84)
            
            # Efficient Numpy Filtering
            classes_scores = data[:, 4:]
            max_scores = np.max(classes_scores, axis=1)
            argmax_classes = np.argmax(classes_scores, axis=1)
            
            mask = max_scores >= min_confidence
            
            valid_scores = max_scores[mask]
            valid_classes = argmax_classes[mask]
            valid_boxes_data = data[mask, 0:4]
            
            for i in range(len(valid_scores)):
                class_id = valid_classes[i]
                
                if class_id not in self.CLASSES:
                    continue
                    
                confidence = valid_scores[i]
                
                # Box is cx, cy, w, h in 640 space
                bx, by, bw, bh = valid_boxes_data[i]
                
                # Scale to image
                left = int((bx - bw/2) * x_factor)
                top = int((by - bh/2) * y_factor)
                width = int(bw * x_factor)
                height = int(bh * y_factor)
                
                boxes.append([left, top, width, height])
                scores.append(float(confidence))
                class_ids.append(class_id)
                
            # NMS
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, min_confidence, 0.45)
                
                # indices is usually flatten list
                if len(indices) > 0:
                    for i in indices.flatten():
                        res = {
                            "box": tuple(boxes[i]),
                            "label": self.CLASSES[class_ids[i]],
                            "conf": scores[i]
                        }
                        results.append(res)
                        
        return results
