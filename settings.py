"""
Управление настройками LPR
"""

import json
import os
import uuid

SETTINGS_FILE = "/opt/lpr/data/settings.json"
CAMERAS_FILE = "/opt/lpr/data/cameras.json"

DEFAULT_SETTINGS = {
    # Telegram (настройте через веб-интерфейс)
    "telegram_token": "",
    "telegram_chat_id": "",
    "telegram_enabled": True,
    "telegram_watchlist_enabled": True,
    # Хранение
    "data_retention_days": 180,
    # Распознавание
    "duplicate_timeout": 300,  # 5 минут - игнорировать дубликаты
    "min_confidence": 30,      # Минимальная уверенность OCR (%)
    # Трекинг
    "min_track_duration": 1.0,      # Мин. длительность трека (сек)
    "min_duration_unknown": 3.0,    # Мин. длительность для нераспознанных (сек)
    "max_frames_ocr": 10,           # Количество лучших кадров для OCR
    "iou_threshold": 0.2,           # Порог IoU трекера
    "max_track_age": 3.0,           # Таймаут потери трека (сек)
    # Маска
    "mask_mode": "bottom"           # Режим проверки маски: center, bottom, all_corners
}

# Камера по умолчанию (для миграции)
DEFAULT_CAMERA = {
    "id": "cam1",
    "name": "Камера 1",
    "url": "rtsp://user:password@camera_ip:554/stream",
    "enabled": True,
    "mask": []
}

def load_settings() -> dict:
    """Загрузка настроек"""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                # Добавляем отсутствующие ключи
                for key, value in DEFAULT_SETTINGS.items():
                    if key not in settings:
                        settings[key] = value
                return settings
        except:
            pass
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict):
    """Сохранение настроек"""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

def get_setting(key: str):
    """Получение одной настройки"""
    settings = load_settings()
    return settings.get(key, DEFAULT_SETTINGS.get(key))


# Маска для зоны распознавания (legacy - для обратной совместимости)
MASK_FILE = "/opt/lpr/data/mask.json"


def load_mask(camera_id: str = None) -> list:
    """
    Загрузка маски для камеры

    Args:
        camera_id: ID камеры (если None - legacy загрузка из mask.json)

    Returns:
        Список точек полигона в нормализованных координатах 0-1
    """
    if camera_id:
        # Новый способ - маска в настройках камеры
        camera = get_camera(camera_id)
        if camera:
            return camera.get('mask', [])
        return []

    # Legacy - старый файл mask.json
    if os.path.exists(MASK_FILE):
        try:
            with open(MASK_FILE, 'r') as f:
                data = json.load(f)
                return data.get('mask', [])
        except:
            pass
    return []


def save_mask(mask: list, camera_id: str = None):
    """
    Сохранение маски

    Args:
        mask: Список точек полигона
        camera_id: ID камеры (если None - legacy сохранение в mask.json)
    """
    if camera_id:
        # Новый способ - сохраняем в настройках камеры
        update_camera(camera_id, {'mask': mask})
    else:
        # Legacy
        with open(MASK_FILE, 'w') as f:
            json.dump({'mask': mask}, f, indent=2)


# ========== Управление камерами ==========

def load_cameras() -> list:
    """Загрузка списка камер"""
    if os.path.exists(CAMERAS_FILE):
        try:
            with open(CAMERAS_FILE, 'r') as f:
                cameras = json.load(f)
                if cameras:
                    return cameras
        except:
            pass

    # Миграция: если есть старая маска, добавляем её к камере по умолчанию
    default = DEFAULT_CAMERA.copy()
    old_mask = load_mask(None)  # legacy загрузка
    if old_mask:
        default['mask'] = old_mask

    return [default]


def save_cameras(cameras: list):
    """Сохранение списка камер"""
    with open(CAMERAS_FILE, 'w') as f:
        json.dump(cameras, f, indent=2, ensure_ascii=False)


def get_camera(camera_id: str) -> dict:
    """Получение камеры по ID"""
    cameras = load_cameras()
    for cam in cameras:
        if cam['id'] == camera_id:
            return cam
    return None


def get_enabled_cameras() -> list:
    """Получение списка включённых камер"""
    return [cam for cam in load_cameras() if cam.get('enabled', True)]


def add_camera(name: str, url: str, enabled: bool = True) -> dict:
    """Добавление новой камеры"""
    cameras = load_cameras()

    new_camera = {
        'id': str(uuid.uuid4())[:8],
        'name': name,
        'url': url,
        'enabled': enabled,
        'mask': []
    }

    cameras.append(new_camera)
    save_cameras(cameras)
    return new_camera


def update_camera(camera_id: str, updates: dict) -> bool:
    """Обновление настроек камеры"""
    cameras = load_cameras()

    for i, cam in enumerate(cameras):
        if cam['id'] == camera_id:
            cameras[i].update(updates)
            save_cameras(cameras)
            return True

    return False


def delete_camera(camera_id: str) -> bool:
    """Удаление камеры"""
    cameras = load_cameras()

    # Нельзя удалить последнюю камеру
    if len(cameras) <= 1:
        return False

    cameras = [cam for cam in cameras if cam['id'] != camera_id]
    save_cameras(cameras)
    return True
