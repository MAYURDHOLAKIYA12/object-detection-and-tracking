import os
from typing import List, Dict, Tuple, Union
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Class mapping according to COCO indices
CLASS_NAMES: Dict[int, str] = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# BGR colors for each class
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 255, 0),      # Person: Green
    2: (255, 0, 0),      # Car: Blue
    3: (0, 165, 255),    # Motorcycle: Orange
    5: (0, 0, 255),      # Bus: Red
    7: (255, 255, 0)     # Truck: Cyan
}

class YOLODetector:
    """
    YOLOv8 Detector wrapper using ultralytics.
    Handles device loading, class filtering, inference, and visualization.
    """
    def __init__(self, model_dir: str = "models", model_name: str = "yolov8n.pt") -> None:
        """
        Initializes the YOLOv8 detector and auto-detects CUDA availability.
        
        Args:
            model_dir (str): Directory where the YOLO model weights are stored.
            model_name (str): Filename of the YOLOv8 model weights.
        """
        # Auto-create models directory
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, model_name)
        
        # Auto-detect CUDA availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[YOLODetector] Loading model onto device: {self.device}")
        
        # Load the model (ultralytics downloads it if not present locally)
        self.model = YOLO(self.model_path)
        
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.4) -> List[Tuple[List[float], float, int]]:
        """
        Runs YOLOv8 object detection on a single frame and filters for specific classes.
        
        Args:
            frame (np.ndarray): The input BGR frame from OpenCV.
            conf_threshold (float): Confidence threshold for detections.
            
        Returns:
            List[Tuple[List[float], float, int]]: A list of detections in the format:
                [([x1, y1, x2, y2], confidence, class_id), ...]
        """
        # Run inference using the selected device
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                # Filter classes: person(0), car(2), motorcycle(3), bus(5), truck(7)
                if class_id in CLASS_NAMES:
                    confidence = float(box.conf[0].item())
                    # Coordinates as [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    detections.append((xyxy, confidence, class_id))
                    
        return detections

    def draw(
        self,
        frame: np.ndarray,
        tracks: Union[List[Tuple[List[float], int, int, float]], np.ndarray],
        show_boxes: bool = True,
        show_ids: bool = True
    ) -> np.ndarray:
        """
        Draws bounding boxes, labels, and tracking IDs on the frame.
        
        Args:
            frame (np.ndarray): The BGR image frame to draw on.
            tracks: List of active tracks or np.ndarray. Each element should represent
                    either tracking tracks: [x1, y1, x2, y2, track_id, class_id, score]
                    or raw detections if tracking is off: [x1, y1, x2, y2, score, class_id].
            show_boxes (bool): Whether to draw bounding boxes and labels.
            show_ids (bool): Whether to display the tracking ID.
            
        Returns:
            np.ndarray: The frame with annotations drawn.
        """
        if not show_boxes and not show_ids:
            return frame
            
        annotated_frame = frame.copy()
        
        for track in tracks:
            # We support both tracking arrays (length 6 or 7) and raw detection lists
            # Tracking array format: [x1, y1, x2, y2, track_id, class_id, score]
            # Detection array format: [x1, y1, x2, y2, score, class_id]
            if len(track) >= 6:
                if len(track) == 7 or (len(track) == 6 and isinstance(track[4], int) and track[4] > 0):
                    # Track format: [x1, y1, x2, y2, track_id, class_id, score]
                    x1, y1, x2, y2 = map(int, track[0:4])
                    track_id = int(track[4])
                    class_id = int(track[5])
                    score = float(track[6])
                else:
                    # Detection format: [x1, y1, x2, y2, score, class_id]
                    x1, y1, x2, y2 = map(int, track[0:4])
                    track_id = None
                    score = float(track[4])
                    class_id = int(track[5])
            else:
                continue
                
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            class_name = CLASS_NAMES.get(class_id, "Unknown")
            
            # 1. Draw bounding box
            if show_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
            # 2. Build text label
            label_parts = []
            if show_ids and track_id is not None:
                label_parts.append(f"ID {track_id}")
            if show_boxes:
                label_parts.append(f"{class_name} {score:.0%}")
                
            label = " | ".join(label_parts)
            
            if label:
                # Calculate size of the label text to create a background rectangle
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                
                # Position label slightly above the box (or inside if too high)
                label_y = max(y1, text_height + 5)
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y - text_height - 5),
                    (x1 + text_width, label_y + baseline - 5),
                    color,
                    cv2.FILLED
                )
                
                # Draw label text (use black text for light colored backgrounds for readability)
                # Let's use black or white based on simple brightness heuristic, or default to white/black
                text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 400 else (255, 255, 255)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                    lineType=cv2.LINE_AA
                )
                
        return annotated_frame
