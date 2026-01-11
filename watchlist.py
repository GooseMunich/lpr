"""
Лист ожидания номеров
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime

DATA_DIR = "/opt/lpr/data"
WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.json")

def _load_watchlist() -> List[Dict]:
    """Загрузить лист ожидания"""
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def _save_watchlist(watchlist: List[Dict]):
    """Сохранить лист ожидания"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)

def get_watchlist() -> List[Dict]:
    """Получить весь лист ожидания"""
    return _load_watchlist()

def add_to_watchlist(pattern: str, delete_on_match: bool = False) -> Dict:
    """Добавить номер/паттерн в лист ожидания"""
    watchlist = _load_watchlist()

    # Нормализуем паттерн (верхний регистр)
    pattern = pattern.upper().strip()

    # Проверяем что такого паттерна ещё нет
    for item in watchlist:
        if item['pattern'] == pattern:
            return item

    new_item = {
        'id': len(watchlist) + 1,
        'pattern': pattern,
        'delete_on_match': delete_on_match,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    watchlist.append(new_item)
    _save_watchlist(watchlist)
    return new_item

def remove_from_watchlist(pattern: str) -> bool:
    """Удалить номер/паттерн из листа ожидания"""
    watchlist = _load_watchlist()
    pattern = pattern.upper().strip()

    new_list = [item for item in watchlist if item['pattern'] != pattern]

    if len(new_list) != len(watchlist):
        _save_watchlist(new_list)
        return True
    return False

def check_watchlist(plate: str) -> Optional[Dict]:
    """
    Проверить, соответствует ли номер какому-либо паттерну в листе ожидания.
    Возвращает совпавший элемент или None.
    """
    watchlist = _load_watchlist()
    plate = plate.upper()

    for item in watchlist:
        pattern = item['pattern']
        # Проверяем вхождение паттерна в номер
        if pattern in plate:
            # Если нужно удалить после совпадения
            if item.get('delete_on_match', False):
                remove_from_watchlist(pattern)
            return item

    return None
