"""
База данных для хранения распознанных номеров
"""

import sqlite3
import os
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple

# Статусы распознавания
STATUS_FULL = 'full'
STATUS_PARTIAL = 'partial'
STATUS_NONE = 'none'

# Путь к БД
DATA_DIR = "/opt/lpr/data"
DB_PATH = f"{DATA_DIR}/plates.db"


def get_connection():
    """Получить соединение с БД"""
    return sqlite3.connect(DB_PATH)


def init_db():
    """Инициализация базы данных"""
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    # Создаём таблицу с новым полем status
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            status TEXT DEFAULT 'full',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Индексы (базовые)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON plates(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_plate ON plates(plate_number)")

    # Миграция: добавляем поле status если его нет
    cursor.execute("PRAGMA table_info(plates)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'status' not in columns:
        print("Миграция БД: добавляем поле status", flush=True)
        cursor.execute("ALTER TABLE plates ADD COLUMN status TEXT DEFAULT 'full'")
        conn.commit()

    # Индекс на status (после миграции)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON plates(status)")

    conn.commit()
    conn.close()


def add_plate(plate_number: str, confidence: float, image_path: str,
              status: str = STATUS_FULL) -> int:
    """
    Добавить распознанный номер

    Args:
        plate_number: номер или "НЕ РАСПОЗНАН"
        confidence: уверенность распознавания
        image_path: путь к изображению
        status: статус распознавания (full/partial/none)

    Returns:
        ID записи
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO plates (plate_number, confidence, image_path, status, timestamp)
           VALUES (?, ?, ?, ?, ?)""",
        (plate_number, confidence, image_path, status,
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    plate_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return plate_id


def get_plates_by_date(target_date: date, status_filter: str = None) -> List[Tuple]:
    """
    Получить номера за дату

    Args:
        target_date: дата
        status_filter: фильтр по статусу (full/partial/none) или None для всех

    Returns:
        Список кортежей (id, plate_number, confidence, image_path, status, timestamp)
    """
    conn = get_connection()
    cursor = conn.cursor()

    if status_filter:
        cursor.execute("""
            SELECT id, plate_number, confidence, image_path, status, timestamp
            FROM plates
            WHERE DATE(timestamp) = ? AND status = ?
            ORDER BY timestamp DESC
        """, (target_date.isoformat(), status_filter))
    else:
        cursor.execute("""
            SELECT id, plate_number, confidence, image_path, status, timestamp
            FROM plates
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        """, (target_date.isoformat(),))

    results = cursor.fetchall()
    conn.close()
    return results


def get_last_plate_time(plate_number: str) -> Optional[datetime]:
    """Получить время последнего распознавания номера"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp FROM plates
        WHERE plate_number = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (plate_number,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return datetime.fromisoformat(result[0])
    return None


def get_recent_plates(seconds: int) -> List[Tuple[str, datetime]]:
    """
    Получить номера за последние N секунд

    Args:
        seconds: количество секунд

    Returns:
        Список кортежей (plate_number, timestamp)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cutoff = datetime.now() - timedelta(seconds=seconds)
    cursor.execute("""
        SELECT plate_number, timestamp FROM plates
        WHERE timestamp > ? AND status != 'none'
        ORDER BY timestamp DESC
    """, (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
    results = cursor.fetchall()
    conn.close()
    return [(row[0], datetime.fromisoformat(row[1])) for row in results]


def search_plates(query: str, status_filter: str = None) -> List[Tuple]:
    """
    Поиск по номеру

    Args:
        query: поисковый запрос
        status_filter: фильтр по статусу

    Returns:
        Список кортежей
    """
    conn = get_connection()
    cursor = conn.cursor()

    if status_filter:
        cursor.execute("""
            SELECT id, plate_number, confidence, image_path, status, timestamp
            FROM plates
            WHERE plate_number LIKE ? AND status = ?
            ORDER BY timestamp DESC
            LIMIT 300
        """, (f"%{query}%", status_filter))
    else:
        cursor.execute("""
            SELECT id, plate_number, confidence, image_path, status, timestamp
            FROM plates
            WHERE plate_number LIKE ?
            ORDER BY timestamp DESC
            LIMIT 300
        """, (f"%{query}%",))

    results = cursor.fetchall()
    conn.close()
    return results


def get_today_count() -> int:
    """Количество распознанных номеров за сегодня"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM plates WHERE DATE(timestamp) = DATE('now')
    """)
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_stats_by_status() -> dict:
    """
    Статистика по статусам за сегодня

    Returns:
        {'full': N, 'partial': N, 'none': N, 'total': N}
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT status, COUNT(*) FROM plates
        WHERE DATE(timestamp) = DATE('now')
        GROUP BY status
    """)
    results = cursor.fetchall()
    conn.close()

    stats = {'full': 0, 'partial': 0, 'none': 0}
    for status, count in results:
        if status in stats:
            stats[status] = count
    stats['total'] = sum(stats.values())
    return stats


def cleanup_old_data(retention_days: int) -> Tuple[int, int]:
    """
    Удаление старых записей и изображений

    Args:
        retention_days: срок хранения в днях

    Returns:
        (deleted_records, deleted_files)
    """
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    # Получаем пути к старым изображениям
    cursor.execute("SELECT image_path FROM plates WHERE timestamp < ?", (cutoff_str,))
    old_images = cursor.fetchall()

    # Удаляем файлы
    deleted_files = 0
    for (image_path,) in old_images:
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                deleted_files += 1
            except:
                pass

    # Удаляем записи из БД
    cursor.execute("DELETE FROM plates WHERE timestamp < ?", (cutoff_str,))
    deleted_records = cursor.rowcount

    conn.commit()
    conn.close()

    if deleted_records > 0:
        print(f"Очистка: удалено {deleted_records} записей, {deleted_files} файлов", flush=True)

    return deleted_records, deleted_files


# Инициализация БД при импорте
init_db()
