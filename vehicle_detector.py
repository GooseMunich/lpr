"""
Детектор транспортных средств на основе YOLOv8
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from settings import load_mask, get_setting, check_mask_invalidated

# COCO классы транспортных средств
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """
    Проверка, находится ли точка внутри полигона (ray casting algorithm)

    Args:
        x, y: координаты точки (0-1)
        polygon: список точек [[x1,y1], [x2,y2], ...]

    Returns:
        True если точка внутри полигона
    """
    n = len(polygon)
    if n < 3:
        return True  # Нет маски = всё разрешено

    inside = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


class VehicleDetector:
    """Детектор транспортных средств на YOLOv8"""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Args:
            model_path: путь к модели YOLO (или имя для автозагрузки)
            confidence: минимальная уверенность детекции
        """
        print(f"Инициализация YOLO детектора: {model_path}", flush=True)
        self.model = YOLO(model_path)
        self.confidence = confidence
        # Маски хранятся по camera_id
        self._masks = {}  # {camera_id: mask}
        self._current_camera_id = None
        print("YOLO детектор инициализирован", flush=True)

    def set_camera(self, camera_id: str):
        """Установка текущей камеры"""
        if camera_id != self._current_camera_id:
            self._current_camera_id = camera_id
            self.reload_mask(camera_id)

    def reload_mask(self, camera_id: str = None):
        """Перезагрузка маски из файла"""
        cam_id = camera_id or self._current_camera_id
        mask = load_mask(cam_id)
        self._masks[cam_id] = mask
        if mask:
            print(f"Маска камеры {cam_id}: {len(mask)} точек", flush=True)
        else:
            print(f"Маска камеры {cam_id} не задана", flush=True)

    def get_mask(self, camera_id: str = None) -> list:
        """Получение маски для камеры"""
        cam_id = camera_id or self._current_camera_id
        # Проверяем, нужно ли перезагрузить маску (была изменена через API)
        if cam_id and check_mask_invalidated(cam_id):
            self.reload_mask(cam_id)
        elif cam_id not in self._masks:
            self.reload_mask(cam_id)
        return self._masks.get(cam_id, [])

    def is_in_mask(self, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int], camera_id: str = None) -> bool:
        """
        Проверка, находится ли bbox внутри маски

        Режимы проверки (настройка mask_mode):
        - center: центр bbox (старое поведение)
        - bottom: нижний центр bbox (где машина касается дороги)
        - all_corners: все 4 угла bbox внутри маски (строгий режим)

        Args:
            bbox: (x1, y1, x2, y2) в пикселях
            frame_shape: (height, width) кадра
            camera_id: ID камеры для получения маски

        Returns:
            True если точка/точки внутри маски или маска не задана
        """
        # Получаем маску для камеры
        mask = self.get_mask(camera_id)

        # Нет маски = всё разрешено
        if not mask or len(mask) < 3:
            return True

        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]

        # Получаем режим проверки
        mask_mode = get_setting("mask_mode")

        if mask_mode == "all_corners":
            # Все 4 угла должны быть внутри маски (строгий режим)
            corners = [
                (x1 / width, y1 / height),  # верхний левый
                (x2 / width, y1 / height),  # верхний правый
                (x1 / width, y2 / height),  # нижний левый
                (x2 / width, y2 / height),  # нижний правый
            ]
            return all(point_in_polygon(cx, cy, mask) for cx, cy in corners)

        elif mask_mode == "bottom":
            # Нижний центр bbox (где машина касается дороги)
            bottom_center_x = ((x1 + x2) / 2) / width
            bottom_center_y = y2 / height
            return point_in_polygon(bottom_center_x, bottom_center_y, mask)

        else:  # center (по умолчанию)
            # Центр bbox
            center_x = ((x1 + x2) / 2) / width
            center_y = ((y1 + y2) / 2) / height
            return point_in_polygon(center_x, center_y, mask)

    def detect(self, frame: np.ndarray, camera_id: str = None) -> List[Dict]:
        """
        Детекция транспортных средств на кадре

        Args:
            frame: BGR изображение (numpy array)
            camera_id: ID камеры для применения маски

        Returns:
            Список детекций: [{'bbox': (x1,y1,x2,y2), 'class': 'car', 'confidence': 0.85}, ...]
        """
        detections = []
        cam_id = camera_id or self._current_camera_id

        # Запуск детекции
        results = self.model(frame, conf=self.confidence, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])

                # Фильтруем только транспортные средства
                if cls_id not in VEHICLE_CLASSES:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))

                # Фильтруем по маске камеры
                if not self.is_in_mask(bbox, frame.shape, cam_id):
                    continue

                detections.append({
                    'bbox': bbox,
                    'class': VEHICLE_CLASSES[cls_id],
                    'confidence': conf
                })

        del results
        return detections

    def get_plate_roi(self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Извлечение области номерного знака из bbox машины
        Номер обычно в нижней трети машины

        Args:
            frame: полный кадр
            vehicle_bbox: (x1, y1, x2, y2) bbox машины

        Returns:
            Обрезанное изображение области номера
        """
        x1, y1, x2, y2 = vehicle_bbox
        height = y2 - y1

        # Берём нижнюю половину машины (там номер)
        plate_y1 = y1 + int(height * 0.4)
        plate_y2 = y2

        # Немного расширяем по бокам
        width = x2 - x1
        plate_x1 = max(0, x1 - int(width * 0.05))
        plate_x2 = min(frame.shape[1], x2 + int(width * 0.05))

        return frame[plate_y1:plate_y2, plate_x1:plate_x2]
