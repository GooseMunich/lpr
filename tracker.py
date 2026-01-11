"""
IoU трекер для отслеживания транспортных средств между кадрами
"""

import time
import threading
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Вычисление Intersection over Union для двух bbox

    Args:
        box1, box2: (x1, y1, x2, y2)

    Returns:
        IoU значение от 0 до 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Пересечение
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Объединение
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedVehicle:
    """Отслеживаемое транспортное средство"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    vehicle_class: str
    first_seen: float
    last_seen: float
    frames: List[Dict] = field(default_factory=list)  # Сохранённые кадры
    plate_detections: List[Dict] = field(default_factory=list)  # Распознанные номера
    best_plate: Optional[Dict] = None  # Лучший распознанный номер

    def add_frame(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], max_frames: int = 50):
        """
        Добавить кадр в буфер

        Args:
            frame: кадр (будет скопирован)
            bbox: bbox машины на этом кадре
            max_frames: максимум кадров в буфере (при 10 FPS: 50 кадров = 5 секунд)
        """
        self.frames.append({
            'frame': frame.copy(),
            'bbox': bbox,
            'timestamp': time.time()
        })
        # Ограничиваем количество кадров
        if len(self.frames) > max_frames:
            self.frames.pop(0)

    def add_plate_detection(self, plate_number: str, confidence: float, bbox: Tuple, frame: np.ndarray):
        """Добавить распознавание номера"""
        detection = {
            'plate_number': plate_number,
            'confidence': confidence,
            'bbox': bbox,
            'frame': frame.copy(),
            'timestamp': time.time()
        }
        self.plate_detections.append(detection)

        # Обновляем лучший номер
        if self.best_plate is None or confidence > self.best_plate['confidence']:
            self.best_plate = detection

    def get_best_frame(self) -> Optional[Dict]:
        """Получить лучший кадр (с номером или самый большой bbox)"""
        if self.best_plate:
            return {
                'frame': self.best_plate['frame'],
                'bbox': self.best_plate['bbox'],
                'plate_number': self.best_plate['plate_number'],
                'confidence': self.best_plate['confidence']
            }

        if not self.frames:
            return None

        # Если номер не распознан - берём кадр с самым большим bbox машины
        best = max(self.frames, key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        return {
            'frame': best['frame'],
            'bbox': best['bbox'],
            'plate_number': None,
            'confidence': 0.0
        }


class VehicleTracker:
    """IoU трекер для транспортных средств"""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: float = 3.0,
        on_track_finished: Optional[Callable[[TrackedVehicle], None]] = None
    ):
        """
        Args:
            iou_threshold: минимальный IoU для сопоставления
            max_age: время в секундах до удаления потерянного трека
            on_track_finished: callback при завершении трека
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.on_track_finished = on_track_finished

        self.tracks: Dict[int, TrackedVehicle] = {}
        self.next_id = 1
        self.lock = threading.Lock()

        # Фоновый поток для проверки таймаутов
        self._running = True
        self._thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._thread.start()

    def update(self, detections: List[Dict], frame: np.ndarray) -> Dict[int, TrackedVehicle]:
        """
        Обновить треки новыми детекциями

        Args:
            detections: список детекций [{'bbox': ..., 'class': ..., 'confidence': ...}]
            frame: текущий кадр

        Returns:
            Словарь активных треков {track_id: TrackedVehicle}
        """
        current_time = time.time()
        matched_tracks = set()
        matched_detections = set()

        with self.lock:
            # Сопоставляем детекции с существующими треками
            for det_idx, detection in enumerate(detections):
                det_bbox = detection['bbox']
                best_iou = 0.0
                best_track_id = None

                for track_id, track in self.tracks.items():
                    if track_id in matched_tracks:
                        continue

                    score = iou(det_bbox, track.bbox)
                    if score > best_iou and score >= self.iou_threshold:
                        best_iou = score
                        best_track_id = track_id

                if best_track_id is not None:
                    # Обновляем существующий трек
                    track = self.tracks[best_track_id]
                    track.bbox = det_bbox
                    track.last_seen = current_time
                    track.add_frame(frame, det_bbox)

                    matched_tracks.add(best_track_id)
                    matched_detections.add(det_idx)

            # Создаём новые треки для несопоставленных детекций
            for det_idx, detection in enumerate(detections):
                if det_idx in matched_detections:
                    continue

                track = TrackedVehicle(
                    track_id=self.next_id,
                    bbox=detection['bbox'],
                    vehicle_class=detection['class'],
                    first_seen=current_time,
                    last_seen=current_time
                )
                track.add_frame(frame, detection['bbox'])

                self.tracks[self.next_id] = track
                self.next_id += 1

            return dict(self.tracks)

    def get_track(self, track_id: int) -> Optional[TrackedVehicle]:
        """Получить трек по ID"""
        with self.lock:
            return self.tracks.get(track_id)

    def add_plate_to_track(self, track_id: int, plate_number: str, confidence: float,
                           bbox: Tuple, frame: np.ndarray):
        """Добавить распознанный номер к треку"""
        with self.lock:
            track = self.tracks.get(track_id)
            if track:
                track.add_plate_detection(plate_number, confidence, bbox, frame)

    def _cleanup_loop(self):
        """Фоновый поток проверки таймаутов"""
        while self._running:
            time.sleep(0.5)
            self._check_timeouts()

    def _check_timeouts(self):
        """Проверить и удалить устаревшие треки"""
        current_time = time.time()
        finished_tracks = []

        with self.lock:
            for track_id, track in list(self.tracks.items()):
                if current_time - track.last_seen > self.max_age:
                    finished_tracks.append(track)
                    del self.tracks[track_id]

        # Вызываем callback для завершённых треков (вне lock)
        for track in finished_tracks:
            duration = track.last_seen - track.first_seen
            plates_count = len(track.plate_detections)
            print(f"Трек #{track.track_id} завершён: {track.vehicle_class}, "
                  f"длительность {duration:.1f}с, номеров: {plates_count}", flush=True)

            if self.on_track_finished:
                try:
                    self.on_track_finished(track)
                except Exception as e:
                    print(f"Ошибка callback для трека #{track.track_id}: {e}", flush=True)

    def stop(self):
        """Остановить трекер"""
        self._running = False
        # Финализируем все треки
        with self.lock:
            tracks = list(self.tracks.values())
            self.tracks.clear()

        for track in tracks:
            if self.on_track_finished:
                try:
                    self.on_track_finished(track)
                except Exception as e:
                    print(f"Ошибка callback при остановке: {e}", flush=True)
