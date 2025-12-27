
import numpy as np
from collections import OrderedDict

class SimpleTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        """
        Args:
            max_disappeared (int): Frames an ID can be missing before Deregistration.
            max_distance (int): Max pixel distance to associate a centroid.
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, detection_data):
        """Register a new object ID."""
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "data": detection_data # (box, label, conf)
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object ID."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracker with list of (box, label, conf).
        Returns: Dictionary of ID -> {centroid, data}
        """
        # If no detections, mark all existing as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Extract centroids from input detections
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        processed_data = [] # Store as tuples for main.py compatibility

        for i, det in enumerate(detections):
            # det is a dictionary {'box':..., 'label':..., 'conf':...}
            box = det['box']
            label = det['label']
            conf = det['conf']
            
            (x, y, w, h) = box
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            input_centroids[i] = (cx, cy)
            
            processed_data.append((box, label, conf))

        # If we have no objects, register all inputs
        if len(self.objects) == 0:
            for i in range(0, len(detections)):
                self.register(input_centroids[i], processed_data[i])
        else:
            # Match existing IDs to new input centroids
            object_ids = list(self.objects.keys())
            object_centroids = [obj["centroid"] for obj in self.objects.values()]

            # Compute distance matrix (Existing vs Input)
            D = self._dist(object_centroids, input_centroids)

            # Find smallest value in each row (closest input for each existing)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Check max distance
                if D[row][col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = {
                    "centroid": input_centroids[col],
                    "data": processed_data[col]
                }
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle disappeared objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Handle new objects (unused inputs)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], processed_data[col])

        return self.objects

    def _dist(self, a, b):
        """Euclidean distance calculator."""
        a = np.array(a)
        b = np.array(b)
        # Expand dims to broadcast
        # Shape A: (N, 2) -> (N, 1, 2)
        # Shape B: (M, 2) -> (1, M, 2)
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
