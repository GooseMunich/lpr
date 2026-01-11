"""
MJPEG видеопоток с отрисовкой распознанных номеров
Поддержка нескольких камер
"""

import cv2
import time
import threading
from typing import Generator, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VideoStream:
    """Менеджер видеопотоков для нескольких камер"""

    def __init__(self):
        self.lock = threading.Lock()
        # Кадры по камерам: {camera_id: jpeg_bytes}
        self._frames: Dict[str, bytes] = {}
        # Детекции по камерам: {camera_id: list}
        self._detections: Dict[str, list] = {}
        # Текущая выбранная камера для отображения
        self._current_camera_id = None

        # Шрифты с поддержкой кириллицы
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except:
            self.font = ImageFont.load_default()
            self.font_small = self.font

    def set_current_camera(self, camera_id: str):
        """Установка текущей камеры для отображения"""
        with self.lock:
            self._current_camera_id = camera_id

    def get_current_camera(self) -> str:
        """Получение ID текущей камеры"""
        with self.lock:
            return self._current_camera_id

    def update_frame(self, frame: np.ndarray, camera_id: str, detections: list = None):
        """
        Обновление кадра с детекциями для камеры

        Args:
            frame: BGR изображение
            camera_id: ID камеры
            detections: список детекций
        """
        if frame is None:
            return

        display_frame = frame.copy()

        # Обновляем детекции
        if detections is not None:
            self._detections[camera_id] = detections
        else:
            detections = self._detections.get(camera_id, [])

        # Конвертируем в PIL для рисования текста
        pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for det in detections:
            if 'bbox' in det:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                plate = det.get('plate_number')
                conf = det.get('confidence', 0)

                # Цвет рамки: зелёный если номер распознан, синий если детекция
                if plate:
                    color = (0, 255, 0)  # Зелёный - номер распознан
                    label = f"{plate} {conf:.0%}"
                else:
                    color = (0, 128, 255)  # Синий - только детекция пластины
                    label = None

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Подпись только если есть номер
                if label:
                    font = self.font_small
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                    padding = 4

                    if y1 - text_h - padding * 2 < 0:
                        text_y = y2
                        bg_y1 = y2
                        bg_y2 = y2 + text_h + padding * 2
                    else:
                        text_y = y1 - text_h - padding
                        bg_y1 = y1 - text_h - padding * 2
                        bg_y2 = y1

                    draw.rectangle([x1, bg_y1, x1 + text_w + padding * 2, bg_y2], fill=color)
                    draw.text((x1 + padding, text_y), label, font=font, fill=(0, 0, 0))

        # Конвертируем обратно в OpenCV
        display_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Уменьшаем для веба
        height, width = display_frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            display_frame = cv2.resize(display_frame, None, fx=scale, fy=scale)

        # Кодируем в JPEG
        _, jpeg = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        with self.lock:
            self._frames[camera_id] = jpeg.tobytes()
            # Если текущая камера не установлена - ставим первую
            if self._current_camera_id is None:
                self._current_camera_id = camera_id

    def get_frame(self, camera_id: str = None) -> bytes:
        """
        Получение кадра для камеры

        Args:
            camera_id: ID камеры (если None - текущая)
        """
        with self.lock:
            cam_id = camera_id or self._current_camera_id
            return self._frames.get(cam_id)

    def generate(self, camera_id: str = None) -> Generator[bytes, None, None]:
        """
        Генератор MJPEG потока

        Args:
            camera_id: ID камеры (если None - текущая выбранная)
        """
        while True:
            with self.lock:
                cam_id = camera_id or self._current_camera_id
            frame = self.get_frame(cam_id)
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)

    def get_cameras_status(self) -> Dict[str, bool]:
        """Получение статуса камер (есть ли кадры)"""
        with self.lock:
            return {cam_id: True for cam_id in self._frames.keys()}


# Глобальный экземпляр
video_stream = VideoStream()
