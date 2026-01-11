import os
from settings import get_setting

# RTSP поток камеры (настраивается через веб-интерфейс в разделе Камеры)
RTSP_URL = "rtsp://user:password@camera_ip:554/stream"

# Telegram (из настроек)
@property
def TELEGRAM_TOKEN():
    return get_setting("telegram_token")

@property  
def TELEGRAM_CHAT_ID():
    return get_setting("telegram_chat_id")

# Настройки распознавания
FRAME_INTERVAL = 0.1
MIN_CONFIDENCE = 0.7

# Динамические настройки
def get_duplicate_timeout():
    return get_setting("duplicate_timeout")

def get_data_retention_days():
    return get_setting("data_retention_days")

# Статические настройки
DUPLICATE_TIMEOUT = 300  # fallback
DATA_DIR = "/opt/lpr/data"
IMAGES_DIR = f"{DATA_DIR}/images"
DB_PATH = f"{DATA_DIR}/plates.db"

WEB_HOST = "0.0.0.0"
WEB_PORT = 8080
LOG_LEVEL = "INFO"

def get_min_confidence():
    return get_setting("min_confidence") / 100.0  # конвертируем в 0.0-1.0
