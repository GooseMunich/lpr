"""
Детектор номерных знаков на основе EasyOCR с трекингом машин

Новый пайплайн:
1. YOLO детектирует машины
2. Трекер отслеживает машины между кадрами
3. EasyOCR распознаёт номера в области каждой машины
4. При завершении трека выбирается лучший результат
5. Три статуса: full (полный номер), partial (частичный), none (не распознан)
"""

import re
import os
import cv2
import easyocr
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import DATA_DIR, IMAGES_DIR, get_duplicate_timeout, get_min_confidence
from database import add_plate, get_last_plate_time, get_recent_plates
from watchlist import check_watchlist
from settings import get_setting
from vehicle_detector import VehicleDetector
from tracker import VehicleTracker, TrackedVehicle
from plate_detector import PlateDetectorYOLO

# Допустимые буквы в российских номерах (12 букв с латинскими аналогами)
VALID_LETTERS = 'АВЕКМНОРСТУХ'
VALID_CHARS = VALID_LETTERS + '0123456789'

# Полный номер: X123XX777 или X123XX77
RU_PLATE_PATTERN = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$')

# Частичный номер: X123XX (без региона) - минимум 6 символов
PARTIAL_PLATE_PATTERN = re.compile(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{1,2}$')

# Таблица замены латиницы на кириллицу
LATIN_TO_CYRILLIC = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х'
}

# Статусы распознавания
STATUS_FULL = 'full'      # Полный номер распознан
STATUS_PARTIAL = 'partial'  # Частично распознан (>=5 символов)
STATUS_NONE = 'none'      # Не распознан


class PlateDetector:
    """Детектор номерных знаков с трекингом машин"""

    def __init__(self, on_result_callback=None):
        """
        Args:
            on_result_callback: функция вызываемая при получении результата
        """
        print("Инициализация системы распознавания...", flush=True)

        # EasyOCR для распознавания текста
        print("Загрузка EasyOCR...", flush=True)
        self.reader = easyocr.Reader(['ru', 'en'], gpu=True)
        # Только допустимые символы для российских номеров
        self.ocr_allowlist = VALID_LETTERS + 'ABEKMHOPCTYX' + '0123456789'  # кириллица + латиница + цифры
        print("EasyOCR загружен", flush=True)

        # YOLO для детекции машин
        self.vehicle_detector = VehicleDetector(confidence=0.4)

        # YOLO для детекции номерных пластин
        self.plate_detector = PlateDetectorYOLO(confidence=0.3)

        # Трекер машин (параметры из настроек)
        self.tracker = VehicleTracker(
            iou_threshold=get_setting("iou_threshold"),
            max_age=get_setting("max_track_age"),
            on_track_finished=self._on_track_finished
        )

        self.on_result = on_result_callback
        self.frame_count = 0
        os.makedirs(IMAGES_DIR, exist_ok=True)

        # Шрифт для отрисовки
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            self.font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            self.font = ImageFont.load_default()
            self.font_small = self.font

        # Таблицы замены символов
        self.digit_to_letter = {'0': 'О', '1': 'Т', '3': 'З', '4': 'Ч', '6': 'Б', '8': 'В', '9': 'Р'}
        self.letter_to_digit = {'О': '0', 'O': '0', 'З': '3', 'Э': '3', 'Ч': '4', 'Б': '6', 'G': '6', 'В': '8', 'B': '8', 'S': '5', 'I': '1', 'L': '1', 'Т': '7'}

        # Результаты для возврата
        self.pending_results = []
        self.results_lock = threading.Lock()

        print("Система распознавания инициализирована", flush=True)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Предобработка кадра для улучшения распознавания"""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        denoised = cv2.bilateralFilter(enhanced, 5, 75, 75)

        return denoised

    def _to_cyrillic(self, text: str) -> str:
        """Заменяем латиницу на кириллицу"""
        result = []
        for c in text.upper():
            result.append(LATIN_TO_CYRILLIC.get(c, c))
        return ''.join(result)

    def _clean_text(self, text: str) -> str:
        """Очистка текста - оставляем только допустимые символы"""
        text = text.upper()
        # Сначала заменяем латиницу на кириллицу
        text = self._to_cyrillic(text)
        # Оставляем только допустимые символы (12 букв + цифры)
        return ''.join(c for c in text if c in VALID_CHARS)

    def _try_normalize(self, text: str) -> Optional[str]:
        """Попытка нормализации текста в формат номера"""
        if len(text) < 5 or len(text) > 9:
            return None

        result = list(text)

        # Нормализация для полного номера (8-9 символов)
        if len(result) >= 8:
            if result[0].isdigit():
                result[0] = self.digit_to_letter.get(result[0], result[0])

            for i in [1, 2, 3]:
                if i < len(result) and result[i].isalpha():
                    result[i] = self.letter_to_digit.get(result[i], result[i])

            for i in [4, 5]:
                if i < len(result) and result[i].isdigit():
                    result[i] = self.digit_to_letter.get(result[i], result[i])

            for i in range(6, len(result)):
                if result[i].isalpha():
                    result[i] = self.letter_to_digit.get(result[i], result[i])

        # Приводим к кириллице
        normalized = self._to_cyrillic(''.join(result))
        return normalized

    def _validate_plate(self, text: str) -> Tuple[bool, str]:
        """
        Проверка и классификация номера

        Формат: X123XX777 (буква, 3 цифры, 2 буквы, 2-3 цифры региона)
        Частичный: X123XX (без региона)

        Returns:
            (is_valid, status): is_valid=True если можно сохранять, status=full/partial/none
        """
        # Полный номер: X123XX77 или X123XX777
        if RU_PLATE_PATTERN.match(text):
            return True, STATUS_FULL

        # Частичный: X123XX (без региона) или X123X (одна буква в конце)
        if PARTIAL_PLATE_PATTERN.match(text):
            return True, STATUS_PARTIAL

        # Дополнительная проверка: номер с частичным регионом X123XX7
        partial_with_region = re.match(r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{1}$', text)
        if partial_with_region:
            return True, STATUS_PARTIAL

        return False, STATUS_NONE

    def _plates_similar(self, plate1: str, plate2: str) -> bool:
        """
        Проверка похожести двух номеров (fuzzy matching)

        Номера считаются похожими если:
        1. Все буквы совпадают И ≥40% всех символов совпадают
        2. ИЛИ ≥70% всех символов совпадают

        Args:
            plate1, plate2: номера для сравнения

        Returns:
            True если номера похожи
        """
        if not plate1 or not plate2:
            return False

        # Если длина сильно отличается - не похожи
        if abs(len(plate1) - len(plate2)) > 2:
            return False

        # Считаем совпадающие символы
        max_len = max(len(plate1), len(plate2))
        min_len = min(len(plate1), len(plate2))

        matches = 0
        for i in range(min_len):
            if plate1[i] == plate2[i]:
                matches += 1

        match_percent = matches / max_len

        # Высокий процент совпадения - точно похожи
        if match_percent >= 0.7:
            return True

        # Проверяем буквы (позиции 0, 4, 5 в формате X123XX)
        # Буквы более стабильны при OCR
        letters1 = [plate1[i] for i in [0, 4, 5] if i < len(plate1) and plate1[i].isalpha()]
        letters2 = [plate2[i] for i in [0, 4, 5] if i < len(plate2) and plate2[i].isalpha()]

        # Если все буквы совпадают и хотя бы 40% общих символов - похожи
        if len(letters1) >= 2 and letters1 == letters2 and match_percent >= 0.4:
            return True

        return False

    def _is_duplicate(self, plate: str) -> bool:
        """
        Проверка на дубликат (с fuzzy matching)

        Номер считается дубликатом если:
        1. Точное совпадение за последние duplicate_timeout секунд
        2. Похожий номер (отличается на 1-2 символа) за последние 30 секунд
        """
        if not plate:
            return False

        timeout = get_duplicate_timeout()

        # 1. Точное совпадение
        last_time = get_last_plate_time(plate)
        if last_time:
            elapsed = (datetime.now() - last_time).total_seconds()
            if elapsed < timeout:
                return True

        # 2. Fuzzy matching - проверяем похожие номера за последние 30 секунд
        recent = get_recent_plates(30)
        for recent_plate, _ in recent:
            if self._plates_similar(plate, recent_plate):
                # Вычисляем процент для лога
                max_len = max(len(plate), len(recent_plate))
                min_len = min(len(plate), len(recent_plate))
                matches = sum(1 for i in range(min_len) if plate[i] == recent_plate[i])
                pct = matches / max_len * 100
                print(f"Fuzzy дубликат: {plate} похож на {recent_plate} ({pct:.0f}%)", flush=True)
                return True

        return False

    def _save_image(self, frame: np.ndarray, vehicle_bbox: Tuple,
                    plate_info: Optional[Dict] = None, status: str = STATUS_NONE) -> Optional[str]:
        """
        Сохранение изображения с рамками

        Args:
            frame: исходный кадр
            vehicle_bbox: bbox машины
            plate_info: информация о номере (bbox, text, confidence)
            status: статус распознавания
        """
        try:
            display_frame = frame.copy()
            pil_image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            vx1, vy1, vx2, vy2 = [int(v) for v in vehicle_bbox]

            # Цвет рамки в зависимости от статуса
            if status == STATUS_FULL:
                color = (0, 255, 0)  # Зелёный
            elif status == STATUS_PARTIAL:
                color = (255, 165, 0)  # Оранжевый
            else:
                color = (255, 0, 0)  # Красный

            # Рамка машины
            draw.rectangle([vx1, vy1, vx2, vy2], outline=color, width=3)

            # Подпись и рамка номера
            if plate_info and plate_info.get('plate_number'):
                plate = plate_info['plate_number']
                conf = plate_info.get('confidence', 0)
                label = f"{plate} {conf:.0%}"

                # Рамка номера и текст над ней
                if plate_info.get('bbox'):
                    px1, py1, px2, py2 = [int(v) for v in plate_info['bbox']]
                    draw.rectangle([px1, py1, px2, py2], outline=(0, 255, 0), width=2)

                    # Текст над рамкой номера
                    text_bbox = draw.textbbox((px1, py1), label, font=self.font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]

                    # Если выходит за верх - рисуем под рамкой
                    if py1 - text_h - 10 < 0:
                        text_y = py2 + 5
                        bg_y1, bg_y2 = py2, py2 + text_h + 10
                    else:
                        text_y = py1 - text_h - 8
                        bg_y1, bg_y2 = py1 - text_h - 10, py1

                    draw.rectangle([px1, bg_y1, px1 + text_w + 10, bg_y2], fill=(0, 255, 0))
                    draw.text((px1 + 5, text_y), label, font=self.font, fill=(0, 0, 0))
                else:
                    # Нет bbox номера - текст над машиной
                    text_bbox = draw.textbbox((vx1, vy1), label, font=self.font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]
                    draw.rectangle([vx1, vy1 - text_h - 10, vx1 + text_w + 10, vy1], fill=color)
                    draw.text((vx1 + 5, vy1 - text_h - 8), label, font=self.font, fill=(0, 0, 0))
            else:
                # Номер не распознан - текст над машиной
                label = "Номер не распознан"
                text_bbox = draw.textbbox((vx1, vy1), label, font=self.font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                draw.rectangle([vx1, vy1 - text_h - 10, vx1 + text_w + 10, vy1], fill=color)
                draw.text((vx1 + 5, vy1 - text_h - 8), label, font=self.font, fill=(0, 0, 0))

            display_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Имя файла
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if plate_info and plate_info.get('plate_number'):
                filename = f"{plate_info['plate_number']}_{timestamp}.jpg"
            else:
                filename = f"UNKNOWN_{timestamp}.jpg"

            filepath = os.path.join(IMAGES_DIR, filename)
            cv2.imwrite(filepath, display_frame)
            return filepath

        except Exception as e:
            print(f"Ошибка сохранения изображения: {e}", flush=True)
            return None

    def _recognize_plate_in_roi(self, frame: np.ndarray, vehicle_bbox: Tuple) -> List[Dict]:
        """
        Распознавание номера в области машины

        Новый пайплайн:
        1. YOLO детектирует номерную пластину в области машины
        2. EasyOCR распознаёт текст на вырезанной пластине

        Args:
            frame: полный кадр
            vehicle_bbox: bbox машины

        Returns:
            Список найденных номеров
        """
        x1, y1, x2, y2 = vehicle_bbox
        plates = []

        # 1. Детектируем номерные пластины в области машины
        plate_detections = self.plate_detector.detect_in_roi(frame, vehicle_bbox)

        if plate_detections:
            # Для каждой найденной пластины запускаем OCR
            for plate_det in plate_detections:
                plate_bbox = plate_det['bbox']

                # Вырезаем пластину с небольшим отступом
                plate_crop = self.plate_detector.crop_plate(frame, plate_bbox, padding=0.1)
                if plate_crop.size == 0:
                    continue

                # Предобработка
                processed_plate = self._preprocess_frame(plate_crop)

                # OCR на пластине
                recognized = self._ocr_on_plate(processed_plate, plate_bbox)
                plates.extend(recognized)

        # 2. Если YOLO не нашёл пластин, используем fallback (нижняя часть машины)
        if not plates:
            height = y2 - y1
            roi_y1 = y1 + int(height * 0.3)
            roi_y2 = y2
            width = x2 - x1
            roi_x1 = max(0, x1 - int(width * 0.05))
            roi_x2 = min(frame.shape[1], x2 + int(width * 0.05))

            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size > 0:
                processed_roi = self._preprocess_frame(roi)
                fallback_plates = self._ocr_on_roi(processed_roi, roi_x1, roi_y1)
                plates.extend(fallback_plates)

        return plates

    def _ocr_on_plate(self, plate_image: np.ndarray, plate_bbox: Tuple) -> List[Dict]:
        """
        OCR на изображении номерной пластины

        Args:
            plate_image: вырезанное изображение пластины
            plate_bbox: bbox пластины в координатах полного кадра

        Returns:
            Список распознанных номеров
        """
        plates = []
        try:
            # Используем allowlist для ограничения символов
            detections = self.reader.readtext(plate_image, allowlist=self.ocr_allowlist)

            for bbox, text, confidence in detections:
                if len(text) < 5:
                    continue

                cleaned = self._clean_text(text)
                if len(cleaned) < 5:
                    continue

                normalized = self._try_normalize(cleaned)
                if normalized is None:
                    continue

                is_valid, status = self._validate_plate(normalized)
                if not is_valid:
                    continue

                if confidence < get_min_confidence():
                    continue

                plates.append({
                    'plate_number': normalized,
                    'confidence': confidence,
                    'bbox': plate_bbox,
                    'status': status
                })

        except Exception as e:
            print(f"Ошибка OCR на пластине: {e}", flush=True)

        return plates

    def _ocr_on_roi(self, roi_image: np.ndarray, offset_x: int, offset_y: int) -> List[Dict]:
        """
        Fallback OCR на ROI (когда YOLO не нашёл пластину)

        Args:
            roi_image: изображение ROI
            offset_x, offset_y: смещение ROI относительно полного кадра

        Returns:
            Список распознанных номеров
        """
        plates = []
        try:
            # Используем allowlist для ограничения символов
            detections = self.reader.readtext(roi_image, allowlist=self.ocr_allowlist)

            for bbox, text, confidence in detections:
                if len(text) < 5:
                    continue

                cleaned = self._clean_text(text)
                if len(cleaned) < 5:
                    continue

                normalized = self._try_normalize(cleaned)
                if normalized is None:
                    continue

                is_valid, status = self._validate_plate(normalized)
                if not is_valid:
                    continue

                if confidence < get_min_confidence():
                    continue

                points = np.array(bbox)
                bx1, by1 = points.min(axis=0)
                bx2, by2 = points.max(axis=0)

                plates.append({
                    'plate_number': normalized,
                    'confidence': confidence,
                    'bbox': (offset_x + bx1, offset_y + by1, offset_x + bx2, offset_y + by2),
                    'status': status
                })

        except Exception as e:
            print(f"Ошибка OCR fallback: {e}", flush=True)

        return plates

    def _select_best_frames(self, track: TrackedVehicle, max_frames: int = 10) -> List[Dict]:
        """
        Отбор лучших кадров для распознавания номера

        Критерий: размер bbox машины (чем больше — тем лучше видно номер)

        Args:
            track: трек машины
            max_frames: максимальное количество кадров для отбора

        Returns:
            Список лучших кадров, отсортированных по размеру bbox
        """
        if not track.frames:
            return []

        # Вычисляем площадь bbox для каждого кадра
        frames_with_area = []
        for frame_data in track.frames:
            bbox = frame_data['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frames_with_area.append({
                'frame': frame_data['frame'],
                'bbox': bbox,
                'area': area,
                'timestamp': frame_data.get('timestamp', 0)
            })

        # Сортируем по площади (от большей к меньшей)
        frames_with_area.sort(key=lambda x: x['area'], reverse=True)

        # Возвращаем топ-N кадров
        return frames_with_area[:max_frames]

    def _recognize_on_best_frames(self, track: TrackedVehicle, max_frames: int = 10) -> Optional[Dict]:
        """
        Распознавание номера на лучших кадрах трека

        Новая логика:
        1. Отбираем лучшие кадры по размеру bbox машины
        2. На каждом кадре ищем номерную пластину (YOLO)
        3. Распознаём текст (EasyOCR)
        4. Выбираем результат с максимальной уверенностью

        Args:
            track: трек машины
            max_frames: количество кадров для обработки

        Returns:
            Dict с лучшим результатом или None
        """
        best_frames = self._select_best_frames(track, max_frames)

        if not best_frames:
            return None

        print(f"Трек #{track.track_id}: отобрано {len(best_frames)} лучших кадров", flush=True)

        best_result = None
        best_confidence = 0

        for i, frame_data in enumerate(best_frames):
            frame = frame_data['frame']
            vehicle_bbox = frame_data['bbox']
            area = frame_data['area']

            # Распознаём номер на этом кадре
            plates = self._recognize_plate_in_roi(frame, vehicle_bbox)

            for plate in plates:
                print(f"  Кадр {i+1} (площадь {area}): {plate['plate_number']} ({plate['confidence']:.0%})", flush=True)

                if plate['confidence'] > best_confidence:
                    best_confidence = plate['confidence']
                    best_result = {
                        'plate_number': plate['plate_number'],
                        'confidence': plate['confidence'],
                        'bbox': plate['bbox'],
                        'status': plate['status'],
                        'frame': frame,
                        'vehicle_bbox': vehicle_bbox
                    }

                    # Если нашли полный номер с высокой уверенностью - можно прекратить
                    if plate['status'] == STATUS_FULL and plate['confidence'] > 0.8:
                        print(f"  -> Найден отличный результат: {plate['plate_number']}", flush=True)
                        return best_result

        if best_result:
            print(f"  -> Лучший результат: {best_result['plate_number']} ({best_confidence:.0%})", flush=True)

        return best_result

    def _on_track_finished(self, track: TrackedVehicle):
        """
        Callback при завершении трека машины

        Новая логика:
        1. Отбираем лучшие кадры по размеру bbox
        2. На каждом кадре ищем номер (YOLO plate + OCR)
        3. Выбираем результат с максимальной уверенностью
        """
        # Параметры из настроек
        min_track_duration = get_setting("min_track_duration")
        max_frames_ocr = get_setting("max_frames_ocr")

        duration = track.last_seen - track.first_seen
        if duration < min_track_duration:
            print(f"Трек #{track.track_id}: слишком короткий ({duration:.1f}с < {min_track_duration}с), пропускаем", flush=True)
            return

        if not track.frames:
            print(f"Трек #{track.track_id}: нет кадров", flush=True)
            return

        print(f"Трек #{track.track_id} завершён ({duration:.1f}с), распознаём номер...", flush=True)

        # Распознаём номер на лучших кадрах
        result = self._recognize_on_best_frames(track, max_frames_ocr)

        if result:
            plate_number = result['plate_number']
            confidence = result['confidence']
            frame = result['frame']
            vehicle_bbox = result['vehicle_bbox']
            status = result['status']
            plate_bbox = result.get('bbox')
        else:
            # Номер не распознан
            # Для нераспознанных требуем более длинный трек (отсекаем ложные короткие)
            min_duration_unknown = get_setting("min_duration_unknown")
            if duration < min_duration_unknown:
                print(f"Трек #{track.track_id}: номер не распознан, трек слишком короткий "
                      f"({duration:.1f}с < {min_duration_unknown}с), пропускаем", flush=True)
                return

            plate_number = None
            confidence = 0
            status = STATUS_NONE
            best_frames = self._select_best_frames(track, 1)
            if best_frames:
                frame = best_frames[0]['frame']
                vehicle_bbox = best_frames[0]['bbox']
            else:
                print(f"Трек #{track.track_id}: нет кадров для сохранения", flush=True)
                return
            plate_bbox = None

        # Проверяем дубликат (только для распознанных номеров)
        if plate_number and self._is_duplicate(plate_number):
            print(f"Трек #{track.track_id}: {plate_number} - дубликат, пропускаем", flush=True)
            return

        # Формируем информацию о номере
        plate_info = None
        if plate_number:
            plate_info = {
                'plate_number': plate_number,
                'confidence': confidence,
                'bbox': plate_bbox
            }

        # Сохраняем изображение
        image_path = self._save_image(frame, vehicle_bbox, plate_info, status)

        # Добавляем в БД
        display_plate = plate_number if plate_number else "НЕ РАСПОЗНАН"
        add_plate(display_plate, confidence, image_path, status)

        # Проверка листа ожидания
        watchlist_match = None
        if plate_number:
            watchlist_match = check_watchlist(plate_number)

        result = {
            'plate_number': plate_number,
            'display_plate': display_plate,
            'confidence': confidence,
            'status': status,
            'image_path': image_path,
            'vehicle_bbox': vehicle_bbox,
            'vehicle_class': track.vehicle_class,
            'watchlist_match': watchlist_match
        }

        # Сохраняем результат
        with self.results_lock:
            self.pending_results.append(result)

        # Логируем
        status_text = {'full': 'ПОЛНЫЙ', 'partial': 'ЧАСТИЧНЫЙ', 'none': 'НЕ РАСПОЗНАН'}
        if watchlist_match:
            print(f"*** НОМЕР (ОЖИДАЕМЫЙ): {display_plate} [{status_text[status]}] "
                  f"(паттерн: {watchlist_match['pattern']}) ***", flush=True)
        else:
            print(f"*** НОМЕР: {display_plate} [{status_text[status]}] "
                  f"({confidence:.0%}) ***", flush=True)

        # Вызываем внешний callback
        if self.on_result:
            try:
                self.on_result(result)
            except Exception as e:
                print(f"Ошибка callback: {e}", flush=True)

    def get_results(self) -> List[Dict]:
        """Получить накопленные результаты"""
        with self.results_lock:
            results = self.pending_results.copy()
            self.pending_results.clear()
        return results

    def process_frame(self, frame: np.ndarray, camera_id: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Обработка кадра - только трекинг, OCR выполняется при завершении трека

        Args:
            frame: BGR изображение
            camera_id: ID камеры для применения маски

        Returns:
            (new_results, current_detections): новые распознанные номера и текущие детекции для видео
        """
        self.frame_count += 1

        if self.frame_count % 30 == 0:
            print(f"Обработано кадров: {self.frame_count}", flush=True)

        all_detections = []

        try:
            # 1. Детектируем машины (с маской для камеры)
            vehicles = self.vehicle_detector.detect(frame, camera_id)

            # 2. Обновляем трекер (кадры накапливаются автоматически)
            tracks = self.tracker.update(vehicles, frame)

            # 3. Для отображения на видео - детектируем только пластины (без OCR)
            for track_id, track in tracks.items():
                # Быстрая детекция пластин для визуализации
                plate_detections = self.plate_detector.detect_in_roi(frame, track.bbox)

                for plate_det in plate_detections:
                    all_detections.append({
                        'plate_number': None,  # OCR не выполняется в реальном времени
                        'confidence': plate_det['confidence'],
                        'bbox': plate_det['bbox'],
                        'status': 'detecting',
                        'track_id': track_id
                    })

        except Exception as e:
            print(f"Ошибка обработки: {e}", flush=True)

        # Возвращаем накопленные результаты
        new_results = self.get_results()
        return new_results, all_detections

    def stop(self):
        """Остановка детектора"""
        self.tracker.stop()
