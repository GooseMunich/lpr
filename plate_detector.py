"""
Детектор номерных пластин на основе YOLOv11
Модель: morsetechlab/yolov11-license-plate-detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


class PlateDetectorYOLO:
    """Детектор номерных пластин на YOLOv11"""

    def __init__(self, model_path: str = "/opt/lpr/plate_detect.pt", confidence: float = 0.3):
        """
        Args:
            model_path: путь к модели YOLO для детекции номеров
            confidence: минимальная уверенность детекции
        """
        print(f"Инициализация детектора номерных пластин: {model_path}", flush=True)
        self.model = YOLO(model_path)
        self.confidence = confidence
        print("Детектор номерных пластин инициализирован", flush=True)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Детекция номерных пластин на кадре

        Args:
            frame: BGR изображение (numpy array)

        Returns:
            Список детекций: [{'bbox': (x1,y1,x2,y2), 'confidence': 0.85}, ...]
        """
        detections = []

        results = self.model(frame, conf=self.confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': conf
                })

        return detections

    def detect_in_roi(self, frame: np.ndarray, roi_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Детекция номерных пластин в заданной области (ROI)

        Args:
            frame: полный кадр
            roi_bbox: область поиска (x1, y1, x2, y2)

        Returns:
            Список детекций с координатами относительно полного кадра
        """
        x1, y1, x2, y2 = roi_bbox

        # Вырезаем ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        # Детектируем в ROI
        local_detections = self.detect(roi)

        # Преобразуем координаты обратно в систему полного кадра
        detections = []
        for det in local_detections:
            lx1, ly1, lx2, ly2 = det['bbox']
            detections.append({
                'bbox': (x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2),
                'confidence': det['confidence']
            })

        return detections

    def crop_plate(self, frame: np.ndarray, plate_bbox: Tuple[int, int, int, int],
                   padding: float = 0.1) -> np.ndarray:
        """
        Вырезать область номерной пластины с небольшим отступом

        Args:
            frame: полный кадр
            plate_bbox: bbox номера (x1, y1, x2, y2)
            padding: процент расширения (0.1 = 10%)

        Returns:
            Обрезанное изображение номерной пластины
        """
        x1, y1, x2, y2 = plate_bbox
        h, w = frame.shape[:2]

        # Добавляем padding
        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)

        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(w, x2 + pw)
        y2 = min(h, y2 + ph)

        return frame[y1:y2, x1:x2]
