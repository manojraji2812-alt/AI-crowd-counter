import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    PERSON_CLASS_ID = 0

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray):
        results = self.model(frame, classes=[self.PERSON_CLASS_ID], verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == self.PERSON_CLASS_ID and conf >= self.confidence:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                    })
        return detections
