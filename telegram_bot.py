"""
Telegram уведомления с поддержкой разных статусов распознавания
"""

import requests
import time
from settings import get_setting

MAX_RETRIES = 3
RETRY_DELAY = 2

# Статусы распознавания
STATUS_FULL = 'full'
STATUS_PARTIAL = 'partial'
STATUS_NONE = 'none'

# Сессия для keep-alive соединений
session = requests.Session()
session.headers.update({'Connection': 'keep-alive'})


def init_bot():
    token = get_setting("telegram_token")
    chat_id = get_setting("telegram_chat_id")
    if token:
        print(f"Telegram настроен (chat_id: {chat_id})", flush=True)
    else:
        print("TELEGRAM_TOKEN не задан", flush=True)


def _send_photo(base_url: str, chat_id: str, message: str, image_path: str) -> bool:
    """Отправка фото с retry"""
    for attempt in range(MAX_RETRIES):
        try:
            with open(image_path, "rb") as photo:
                resp = session.post(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id, "caption": message, "parse_mode": "Markdown"},
                    files={"photo": photo},
                    timeout=60
                )
            if resp.ok:
                return True
            print(f"Telegram API error: {resp.text}", flush=True)
        except FileNotFoundError:
            # Файл не найден - отправим текст
            return _send_text(base_url, chat_id, message)
        except Exception as e:
            print(f"Telegram error (попытка {attempt + 1}/{MAX_RETRIES}): {e}", flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return False


def _send_text(base_url: str, chat_id: str, message: str) -> bool:
    """Отправка текста с retry"""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.post(
                f"{base_url}/sendMessage",
                data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=30
            )
            if resp.ok:
                return True
        except Exception as e:
            print(f"Telegram text error (попытка {attempt + 1}/{MAX_RETRIES}): {e}", flush=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return False


async def send_plate_notification(plate: str, confidence: float, image_path: str = None,
                                   status: str = STATUS_FULL, vehicle_class: str = None):
    """
    Отправка уведомления о распознанном номере

    Args:
        plate: номер или "НЕ РАСПОЗНАН"
        confidence: уверенность
        image_path: путь к изображению
        status: статус распознавания (full/partial/none)
        vehicle_class: тип ТС (car/truck/bus/motorcycle)
    """
    # Проверка включены ли уведомления
    if not get_setting("telegram_enabled"):
        return

    token = get_setting("telegram_token")
    chat_id = get_setting("telegram_chat_id")

    if not token or not chat_id:
        return

    base_url = f"https://api.telegram.org/bot{token}"

    # Формируем сообщение в зависимости от статуса
    if status == STATUS_FULL:
        message = f"🟢 Номер: *{plate}*\nУверенность: {confidence:.0%}"
    elif status == STATUS_PARTIAL:
        message = f"🟡 Номер: *{plate}*\nУверенность: {confidence:.0%}"
    else:  # STATUS_NONE
        message = "🔴 Номер не распознан"

    if image_path:
        if _send_photo(base_url, chat_id, message, image_path):
            print(f"Telegram OK: {plate} [{status}]", flush=True)
        else:
            print(f"Telegram FAILED: {plate}", flush=True)
    else:
        _send_text(base_url, chat_id, message)


async def send_watchlist_notification(plate: str, pattern: str, confidence: float,
                                       image_path: str = None, status: str = STATUS_FULL):
    """Отправка уведомления об ожидаемом номере"""
    # Проверка включены ли уведомления об ожидаемых номерах
    if not get_setting("telegram_watchlist_enabled"):
        return

    token = get_setting("telegram_token")
    chat_id = get_setting("telegram_chat_id")

    if not token or not chat_id:
        return

    base_url = f"https://api.telegram.org/bot{token}"

    # Статус распознавания в тексте
    if status == STATUS_FULL:
        status_text = ""
    elif status == STATUS_PARTIAL:
        status_text = " _(частично)_"
    else:
        status_text = " _(не распознан)_"

    message = (f"⚠️ *ОЖИДАЕМЫЙ НОМЕР!*\n\n"
               f"🚗 Номер: *{plate}*{status_text}\n"
               f"🔍 Паттерн: `{pattern}`\n"
               f"Уверенность: {confidence:.0%}")

    if image_path:
        if _send_photo(base_url, chat_id, message, image_path):
            print(f"Telegram WATCHLIST OK: {plate} (pattern: {pattern})", flush=True)
        else:
            print(f"Telegram WATCHLIST FAILED: {plate}", flush=True)
    else:
        if _send_text(base_url, chat_id, message):
            print(f"Telegram WATCHLIST OK: {plate} (pattern: {pattern})", flush=True)
        else:
            print(f"Telegram WATCHLIST FAILED: {plate}", flush=True)
