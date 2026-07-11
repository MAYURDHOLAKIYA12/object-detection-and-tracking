from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) between two sets of bounding boxes.
    
    Args:
        bb_test (np.ndarray): Predicted bounding boxes of shape (N, 4).
        bb_gt (np.ndarray): Detected bounding boxes of shape (M, 4).
        
    Returns:
        np.ndarray: IoU matrix of shape (N, M).
    """
    # Expand dims for broadcasting
    bb_test = np.expand_dims(bb_test, 1)  # Shape (N, 1, 4)
    bb_gt = np.expand_dims(bb_gt, 0)      # Shape (1, M, 4)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    
    iou_matrix = wh / (area_test + area_gt - wh + 1e-6)
    return iou_matrix

def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Converts a bounding box in format [x1, y1, x2, y2] to measurement vector z
    in format [x_c, y_c, s, r]^T where x_c, y_c is the center of the box,
    s is the scale/area, and r is the aspect ratio.
    
    Args:
        bbox (np.ndarray): Bounding box of shape (4,).
        
    Returns:
        np.ndarray: State/measurement vector of shape (4, 1).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x_c = bbox[0] + w / 2.0
    y_c = bbox[1] + h / 2.0
    s = w * h
    r = w / (float(h) + 1e-6)
    return np.array([x_c, y_c, s, r]).reshape((4, 1))

def z_to_bbox(x: np.ndarray) -> np.ndarray:
    """
    Converts a state vector x of format [x_c, y_c, s, r, ...] to bounding box
    in format [x1, y1, x2, y2].
    
    Args:
        x (np.ndarray): State vector of shape (7, 1).
        
    Returns:
        np.ndarray: Bounding box of shape (1, 4).
    """
    x_c, y_c, s, r = x[0, 0], x[1, 0], x[2, 0], x[3, 0]
    w = np.sqrt(max(0.0, s * r))
    h = np.sqrt(max(0.0, s / (r + 1e-6)))
    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0
    return np.array([x1, y1, x2, y2]).reshape((1, 4))

class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    Uses Kalman Filter to predict state trajectories.
    """
    count = 0

    def __init__(self, bbox: np.ndarray, class_id: int, score: float) -> None:
        """
        Initializes a tracker using initial bounding box.
        
        Args:
            bbox (np.ndarray): Initial bounding box coordinates [x1, y1, x2, y2].
            class_id (int): Coco class ID of the object.
            score (float): Confidence score of the detection.
        """
        # Define constant velocity motion model
        # State vector: [x_c, y_c, s, r, dx_c, dy_c, ds]^T
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix F
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix H
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Covariances initialization
        self.kf.R = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0, 10.0]
        ])
        
        self.kf.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0])
        self.kf.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001])
        
        # Initialize state with detection
        self.kf.x[:4] = bbox_to_z(bbox)
        
        self.time_since_update = 0
        
        # Tracking metadata
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        
        self.history = []
        self.hits = 1
        self.age = 0
        self.class_id = class_id
        self.score = score

    def update(self, bbox: np.ndarray, class_id: int, score: float) -> None:
        """
        Updates the state vector with a new bounding box detection.
        
        Args:
            bbox (np.ndarray): Bounding box coordinates [x1, y1, x2, y2].
            class_id (int): The associated class ID of the matched detection.
            score (float): The confidence score of the matched detection.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.class_id = class_id
        self.score = score
        self.kf.update(bbox_to_z(bbox))

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        
        Returns:
            np.ndarray: Bounding box coordinates of shape (1, 4).
        """
        # Ensure scale/area velocity doesn't make scale go negative
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] *= 0.0
            
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        self.history.append(z_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate.
        
        Returns:
            np.ndarray: Bounding box coordinates of shape (4,).
        """
        return z_to_bbox(self.kf.x)[0]

class Sort:
    """
    Simple Online and Realtime Tracker (SORT) implementation.
    Manages multiple KalmanBoxTracker instances and performs data association.
    """
    def __init__(self, max_age: int = 30, min_hits: int = 2, iou_threshold: float = 0.3) -> None:
        """
        Initializes the SORT tracker tracker.
        
        Args:
            max_age (int): Maximum frames to keep dead trackers.
            min_hits (int): Minimum frames an object must be hit before being reported.
            iou_threshold (float): IoU threshold for matching detections to trackers.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Updates the tracker list and returns active tracks.
        
        Args:
            dets (np.ndarray): Detections in the current frame.
                               Shape (N, 6) where each row is:
                               [x1, y1, x2, y2, score, class_id]
                               
        Returns:
            np.ndarray: Active tracks of shape (M, 7) where each row is:
                        [x1, y1, x2, y2, track_id, class_id, score]
        """
        self.frame_count += 1
        
        # 1. Get predicted positions from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove trackers with NaN states
        trks = np.delete(trks, to_del, axis=0)
        for index in sorted(to_del, reverse=True):
            self.trackers.pop(index)
            
        # If no detections, update status of trackers and return empty list
        if dets is None or len(dets) == 0:
            # Check for dead trackers
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            ret = []
            for trk in self.trackers:
                if trk.time_since_update < 1 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                    ret.append(np.concatenate((trk.get_state(), [trk.id, trk.class_id, trk.score])))
            if len(ret) > 0:
                return np.array(ret)
            return np.empty((0, 7))

        # 2. Perform Association (IoU & Hungarian Algorithm)
        # Split detections to bounding boxes
        det_bboxes = dets[:, 0:4]
        iou_matrix = iou_batch(det_bboxes, trks)
        
        if min(iou_matrix.shape) > 0:
            # We negate IoU to transform the maximization problem into minimization (Hungarian)
            cost_matrix = -iou_matrix
            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(matched_indices[0], matched_indices[1])))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
            
        # Parse unmatched detections, unmatched trackers, and matched pairs
        unmatched_detections = []
        for d, det in enumerate(dets):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
                
        # Filter matches below IoU threshold
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m)
                
        # 3. Update matched trackers
        for m in matches:
            det_idx, trk_idx = m[0], m[1]
            bbox = dets[det_idx, 0:4]
            score = dets[det_idx, 4]
            class_id = int(dets[det_idx, 5])
            self.trackers[trk_idx].update(bbox, class_id, score)
            
        # 4. Initialize new trackers for unmatched detections
        for d in unmatched_detections:
            bbox = dets[d, 0:4]
            score = dets[d, 4]
            class_id = int(dets[d, 5])
            trk = KalmanBoxTracker(bbox, class_id, score)
            self.trackers.append(trk)
            
        # 5. Retrieve active tracks to return and prune dead trackers
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # If the track was updated this frame and meets min hits criteria
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                # Format: [x1, y1, x2, y2, track_id, class_id, score]
                ret.append(np.concatenate((d, [trk.id, trk.class_id, trk.score])))
            i -= 1
            # Prune dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 7))

    def get_lifetime_count(self) -> int:
        """
        Returns the lifetime count of unique tracked objects assigned since start.
        
        Returns:
            int: The maximum assigned track ID.
        """
        return KalmanBoxTracker.count
