# LPR - License Plate Recognition System v3.0

Система распознавания российских автомобильных номеров с двухступенчатой детекцией, трекингом и поддержкой нескольких камер.

![Пример работы системы](screenshot.png)

## Возможности

- **Двухступенчатая детекция**: YOLOv8 (машины) + YOLOv11 (номерные пластины) + EasyOCR
- **GPU-ускорение**: NVIDIA CUDA для всех этапов обработки
- **Трекинг машин**: IoU-трекер для отслеживания между кадрами
- **Мульти-камера**: поддержка нескольких RTSP камер
- **Веб-интерфейс**: просмотр номеров, видеопоток, настройки
- **Telegram уведомления**: оповещения о распознанных номерах
- **Лист ожидания**: отслеживание конкретных номеров

## Требования

- Ubuntu 22.04/24.04
- NVIDIA GPU с поддержкой CUDA
- Python 3.10+
- 8GB+ RAM

## Установка

### 1. NVIDIA драйверы и CUDA

```bash
# Установить драйвер
sudo apt install -y nvidia-driver-535
sudo reboot

# Проверить
nvidia-smi
```

### 2. Создание окружения

```bash
# Создать директории
sudo mkdir -p /opt/lpr
sudo chown $USER:$USER /opt/lpr

# Клонировать репозиторий
git clone https://github.com/YOUR_USERNAME/lpr.git /opt/lpr

# Виртуальное окружение
cd /opt/lpr
python3 -m venv venv
source venv/bin/activate

# PyTorch с CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Зависимости
pip install easyocr ultralytics
pip install fastapi uvicorn jinja2 python-multipart
pip install Pillow opencv-python requests
```

### 3. Модель детекции номерных пластин

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download('morsetechlab/yolov11-license-plate-detection', 'license-plate-finetune-v1n.pt')
shutil.copy(path, '/opt/lpr/plate_detect.pt')
print('Модель скачана')
"
```

### 4. Создание директории данных

```bash
mkdir -p /opt/lpr/data/images
```

### 5. Systemd сервис

```bash
sudo tee /etc/systemd/system/lpr.service << 'EOF'
[Unit]
Description=LPR - License Plate Recognition System
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/opt/lpr
ExecStart=/opt/lpr/venv/bin/python main.py
Restart=always
RestartSec=5
MemoryMax=8G

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable lpr
sudo systemctl start lpr
```

## Настройка

### Веб-интерфейс

После запуска откройте http://YOUR_IP:8080

| Страница | URL | Описание |
|----------|-----|----------|
| Главная | / | Список распознанных номеров |
| Видео | /video | Живой поток с детекциями |
| Ожидание | /watchlist | Лист ожидания номеров |
| Настройки | /settings | Telegram, камеры, параметры |

### Камеры

На странице /settings добавьте RTSP камеры:
```
rtsp://user:password@camera_ip:554/stream
```

### Telegram

1. Создайте бота через @BotFather
2. Получите токен
3. Добавьте бота в группу/чат
4. Получите chat_id
5. Введите данные на странице /settings

### Маска зоны распознавания

На странице /video можно нарисовать полигон для ограничения области детекции (исключение припаркованных машин).

## Пайплайн распознавания

```
RTSP поток (10 FPS)
    ↓
[YOLO: детекция машин]
    ↓
[IoU-трекер: накопление до 50 кадров]
    ↓
[Машина уехала (трек завершён)]
    ↓
[Отбор лучших кадров по размеру bbox]
    ↓
[На каждом: YOLO plate → EasyOCR]
    ↓
[Выбор результата с максимальной уверенностью]
    ↓
[Сохранение: БД + фото + Telegram]
```

## Формат российских номеров

Формат: `X123XX777` (буква, 3 цифры, 2 буквы, регион 2-3 цифры)

Допустимые буквы: А, В, Е, К, М, Н, О, Р, С, Т, У, Х

| Статус | Описание |
|--------|----------|
| full | Полный номер (8-9 символов) |
| partial | Частичный номер (≥5 символов) |
| none | Номер не распознан |

## Управление

```bash
# Статус
sudo systemctl status lpr

# Запустить / Остановить / Перезапустить
sudo systemctl start lpr
sudo systemctl stop lpr
sudo systemctl restart lpr

# Логи
journalctl -u lpr -f
tail -f /opt/lpr/data/lpr.log
```

## Структура проекта

```
/opt/lpr/
├── main.py              # Точка входа
├── capture.py           # Захват RTSP потока
├── vehicle_detector.py  # Детекция машин (YOLOv8)
├── plate_detector.py    # Детекция номерных пластин (YOLOv11)
├── plate_detect.pt      # Модель детекции пластин
├── tracker.py           # IoU трекер машин
├── detector.py          # Распознавание номеров (EasyOCR)
├── database.py          # SQLite операции
├── web.py               # FastAPI веб-сервер
├── video_stream.py      # MJPEG видеопоток
├── telegram_bot.py      # Telegram уведомления
├── watchlist.py         # Лист ожидания
├── settings.py          # Управление настройками
├── config.py            # Конфигурация
├── templates/           # HTML шаблоны
├── static/              # Статические файлы
└── data/                # Данные (БД, изображения, настройки)
```

## Лицензия

MIT License
