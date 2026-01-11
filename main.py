#!/usr/bin/env python3
"""
LPR System - License Plate Recognition с трекингом машин
Поддержка нескольких камер

Пайплайн:
1. YOLO детектирует машины на кадре
2. Трекер отслеживает машины между кадрами
3. EasyOCR распознаёт номера в области каждой машины
4. При завершении трека выбирается лучший результат
5. Результат сохраняется в БД и отправляется в Telegram
"""

import asyncio
import logging
import signal
import time
from datetime import datetime
from typing import Dict

from config import LOG_LEVEL, WEB_HOST, WEB_PORT, get_data_retention_days, FRAME_INTERVAL
from database import init_db, cleanup_old_data
from capture import RTSPCapture
from detector import PlateDetector
from web import app
from telegram_bot import init_bot, send_plate_notification, send_watchlist_notification
from video_stream import video_stream
from settings import load_cameras, get_enabled_cameras

import uvicorn
import threading

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/lpr/data/lpr.log')
    ]
)
logger = logging.getLogger(__name__)

running = True
detector = None
last_cleanup = 0
captures: Dict[str, RTSPCapture] = {}  # {camera_id: RTSPCapture}


def signal_handler(signum, frame):
    global running
    logger.info(f"Получен сигнал {signum}, завершаю работу...")
    running = False


def send_telegram(result: dict):
    """Отправка уведомления в Telegram"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        plate = result.get('display_plate', 'НЕ РАСПОЗНАН')
        confidence = result.get('confidence', 0)
        image_path = result.get('image_path')
        status = result.get('status', 'none')
        vehicle_class = result.get('vehicle_class')
        watchlist_match = result.get('watchlist_match')

        if watchlist_match:
            loop.run_until_complete(
                send_watchlist_notification(
                    plate,
                    watchlist_match['pattern'],
                    confidence,
                    image_path,
                    status
                )
            )
        else:
            loop.run_until_complete(
                send_plate_notification(
                    plate,
                    confidence,
                    image_path,
                    status,
                    vehicle_class
                )
            )

        loop.close()
    except Exception as e:
        print(f"Ошибка Telegram: {e}", flush=True)


def create_frame_callback(camera_id: str, camera_name: str):
    """Создание callback для обработки кадров с конкретной камеры"""
    def on_frame(frame):
        global detector
        if detector is None:
            return

        try:
            # Устанавливаем текущую камеру в детекторе
            detector.vehicle_detector.set_camera(camera_id)

            # Обрабатываем кадр с указанием камеры
            new_plates, all_plates = detector.process_frame(frame, camera_id)

            # Обновляем видеопоток для этой камеры
            video_stream.update_frame(frame, camera_id, all_plates)

            for result in new_plates:
                plate_number = result.get('plate_number') or result.get('display_plate', 'НЕ РАСПОЗНАН')
                confidence = result.get('confidence', 0)
                status = result.get('status', 'none')

                status_text = {'full': 'полный', 'partial': 'частичный', 'none': 'не распознан'}
                logger.info(f"[{camera_name}] Результат: {plate_number} (статус: {status_text.get(status, status)}, "
                           f"уверенность: {confidence:.0%})")

                # Добавляем camera_id к результату
                result['camera_id'] = camera_id
                result['camera_name'] = camera_name

                # Отправляем в Telegram
                telegram_thread = threading.Thread(
                    target=send_telegram,
                    args=(result,),
                    daemon=True
                )
                telegram_thread.start()

        except Exception as e:
            logger.error(f"[{camera_name}] Ошибка обработки кадра: {e}")

    return on_frame


def run_web_server():
    config = uvicorn.Config(app, host=WEB_HOST, port=WEB_PORT, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def run_cleanup():
    """Очистка старых данных раз в час"""
    global last_cleanup
    current_time = time.time()

    if current_time - last_cleanup > 3600:
        last_cleanup = current_time
        try:
            retention_days = get_data_retention_days()
            cleanup_old_data(retention_days)
        except Exception as e:
            print(f"Ошибка очистки: {e}", flush=True)


def start_cameras():
    """Запуск захвата для всех включённых камер"""
    global captures

    cameras = get_enabled_cameras()

    if not cameras:
        logger.warning("Нет включённых камер!")
        return

    for cam in cameras:
        camera_id = cam['id']
        camera_name = cam.get('name', camera_id)
        url = cam['url']

        logger.info(f"Запуск камеры '{camera_name}': {url}")

        callback = create_frame_callback(camera_id, camera_name)
        capture = RTSPCapture(url, callback)
        capture.start()
        captures[camera_id] = capture

    logger.info(f"Запущено камер: {len(captures)}")


def stop_cameras():
    """Остановка всех камер"""
    global captures
    for camera_id, capture in captures.items():
        logger.info(f"Остановка камеры {camera_id}")
        capture.stop()
    captures.clear()


def main():
    global running, detector, last_cleanup

    logger.info("=" * 50)
    logger.info("Запуск системы LPR v3.0 (мульти-камера)")
    logger.info(f"Время: {datetime.now()}")
    logger.info("=" * 50)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Инициализация базы данных...")
    init_db()

    logger.info("Инициализация Telegram...")
    init_bot()

    logger.info("Инициализация детектора (YOLO + EasyOCR + трекер)...")
    detector = PlateDetector()

    logger.info(f"Запуск веб-сервера на http://{WEB_HOST}:{WEB_PORT}")
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()

    # Загружаем и запускаем камеры
    logger.info("Загрузка камер...")
    start_cameras()

    # Первая очистка при старте
    last_cleanup = time.time()
    run_cleanup()

    logger.info("Система LPR запущена")
    logger.info("Пайплайн: YOLO -> Трекер -> EasyOCR -> Консенсус")

    try:
        while running:
            time.sleep(1)
            run_cleanup()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        detector.stop()
        stop_cameras()
        logger.info("Система LPR остановлена")


if __name__ == "__main__":
    main()
