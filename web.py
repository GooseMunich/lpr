from fastapi import FastAPI, Request, Query, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import date, datetime
import config
import database
from video_stream import video_stream
from settings import (
    load_settings, save_settings, load_mask, save_mask,
    load_cameras, save_cameras, get_camera, add_camera, update_camera, delete_camera
)
from watchlist import get_watchlist, add_to_watchlist, remove_from_watchlist

app = FastAPI(title="LPR System v2.0")

app.mount("/static", StaticFiles(directory="/opt/lpr/static"), name="static")
app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
templates = Jinja2Templates(directory="/opt/lpr/templates")


@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    date_filter: str = Query(None, alias="date"),
    search: str = Query(None),
    status: str = Query(None)
):
    """Главная страница со списком номеров"""
    # Получаем статистику по статусам
    stats = database.get_stats_by_status()

    if search:
        plates = database.search_plates(search, status)
        title = f"Поиск: {search}"
    elif date_filter:
        try:
            filter_date = datetime.strptime(date_filter, "%Y-%m-%d").date()
        except ValueError:
            filter_date = date.today()
        plates = database.get_plates_by_date(filter_date, status)
        title = f"Номера за {filter_date}"
    else:
        plates = database.get_plates_by_date(date.today(), status)
        title = f"Номера за сегодня ({date.today()})"

    # Добавляем информацию о фильтре статуса
    status_names = {
        'full': 'Полные',
        'partial': 'Частичные',
        'none': 'Не распознанные'
    }
    if status:
        title += f" - {status_names.get(status, status)}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "plates": plates,
        "title": title,
        "today": date.today().isoformat(),
        "search_query": search or "",
        "date_filter": date_filter or date.today().isoformat(),
        "status_filter": status or "",
        "stats": stats
    })


@app.get("/video", response_class=HTMLResponse)
async def video_page(request: Request, camera: str = None):
    """Страница видеопотока с выбором камеры"""
    cameras = load_cameras()
    # Если камера не указана - берём первую
    current_camera = camera or (cameras[0]['id'] if cameras else None)
    if current_camera:
        video_stream.set_current_camera(current_camera)
    return templates.TemplateResponse("video.html", {
        "request": request,
        "cameras": cameras,
        "current_camera": current_camera
    })


@app.get("/video_feed")
async def video_feed(camera: str = None):
    """MJPEG видеопоток для выбранной камеры"""
    return StreamingResponse(
        video_stream.generate(camera),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, saved: bool = False):
    settings = load_settings()
    cameras = load_cameras()
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "settings": settings,
        "cameras": cameras,
        "saved": saved
    })


@app.post("/settings")
async def settings_save(
    telegram_token: str = Form(""),
    telegram_chat_id: str = Form(""),
    telegram_enabled: str = Form(None),
    telegram_watchlist_enabled: str = Form(None),
    duplicate_timeout: int = Form(5),
    data_retention_days: int = Form(180),
    min_confidence: int = Form(30),
    max_frames_ocr: int = Form(10),
    min_track_duration: float = Form(1.0),
    min_duration_unknown: float = Form(3.0),
    max_track_age: float = Form(3.0),
    iou_threshold: float = Form(0.2),
    mask_mode: str = Form("bottom")
):
    settings = {
        "telegram_token": telegram_token,
        "telegram_chat_id": telegram_chat_id,
        "telegram_enabled": telegram_enabled == "1",
        "telegram_watchlist_enabled": telegram_watchlist_enabled == "1",
        "duplicate_timeout": duplicate_timeout * 60,  # в секунды
        "data_retention_days": data_retention_days,
        "min_confidence": min_confidence,
        "max_frames_ocr": max_frames_ocr,
        "min_track_duration": min_track_duration,
        "min_duration_unknown": min_duration_unknown,
        "max_track_age": max_track_age,
        "iou_threshold": iou_threshold,
        "mask_mode": mask_mode
    }
    save_settings(settings)
    return RedirectResponse(url="/settings?saved=true", status_code=303)


# Лист ожидания
@app.get("/watchlist", response_class=HTMLResponse)
async def watchlist_page(request: Request, added: str = None, removed: str = None):
    watchlist = get_watchlist()
    return templates.TemplateResponse("watchlist.html", {
        "request": request,
        "watchlist": watchlist,
        "added": added,
        "removed": removed
    })


@app.post("/watchlist/add")
async def watchlist_add(
    pattern: str = Form(...),
    delete_on_match: str = Form(None)
):
    delete_flag = delete_on_match == "1"
    add_to_watchlist(pattern, delete_flag)
    return RedirectResponse(url=f"/watchlist?added={pattern}", status_code=303)


@app.post("/watchlist/remove")
async def watchlist_remove(pattern: str = Form(...)):
    remove_from_watchlist(pattern)
    return RedirectResponse(url=f"/watchlist?removed={pattern}", status_code=303)


@app.get("/api/stats")
async def stats():
    """API: статистика распознавания"""
    stats = database.get_stats_by_status()
    return stats


@app.get("/restart")
async def restart_service():
    """Перезапуск сервиса"""
    import subprocess
    import threading

    def do_restart():
        import time
        time.sleep(1)
        subprocess.run(["sudo", "systemctl", "restart", "lpr"])

    threading.Thread(target=do_restart, daemon=True).start()

    return HTMLResponse('''
        <html>
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="5;url=/settings">
            <style>
                body { font-family: sans-serif; background: #1a1a2e; color: #eee;
                       display: flex; justify-content: center; align-items: center;
                       height: 100vh; margin: 0; }
                .msg { text-align: center; }
                .spinner { font-size: 3rem; animation: spin 1s linear infinite; }
                @keyframes spin { 100% { transform: rotate(360deg); } }
            </style>
        </head>
        <body>
            <div class="msg">
                <div class="spinner">⚙️</div>
                <h2>Перезапуск сервиса...</h2>
                <p>Страница обновится автоматически</p>
            </div>
        </body>
        </html>
    ''')


@app.get("/api/gpu")
async def gpu_stats():
    """Статистика GPU"""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,fan.speed,memory.used,memory.total,power.draw,name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'gpu_util': int(parts[0]),
                'temp': int(parts[1]),
                'fan': int(parts[2]) if parts[2] != '[N/A]' else None,
                'mem_used': int(parts[3]),
                'mem_total': int(parts[4]),
                'power': float(parts[5]) if parts[5] != '[N/A]' else None,
                'name': parts[6]
            }
    except Exception as e:
        return {'error': str(e)}
    return {'error': 'nvidia-smi failed'}


@app.get("/api/mask")
async def get_mask_api(camera: str = None):
    """Получить маску зоны распознавания для камеры"""
    mask = load_mask(camera)
    return {'mask': mask, 'camera_id': camera}


@app.post("/api/mask")
async def set_mask_api(request: Request):
    """Сохранить маску зоны распознавания для камеры"""
    data = await request.json()
    mask = data.get('mask', [])
    camera_id = data.get('camera_id')
    save_mask(mask, camera_id)
    return {'status': 'ok', 'points': len(mask), 'camera_id': camera_id}


# ========== API для камер ==========

@app.get("/api/cameras")
async def get_cameras_api():
    """Получить список всех камер"""
    cameras = load_cameras()
    return {'cameras': cameras}


@app.post("/api/cameras")
async def add_camera_api(request: Request):
    """Добавить новую камеру"""
    data = await request.json()
    name = data.get('name', 'Новая камера')
    url = data.get('url', '')
    enabled = data.get('enabled', True)

    if not url:
        return JSONResponse({'error': 'URL обязателен'}, status_code=400)

    camera = add_camera(name, url, enabled)
    return {'status': 'ok', 'camera': camera}


@app.put("/api/cameras/{camera_id}")
async def update_camera_api(camera_id: str, request: Request):
    """Обновить настройки камеры"""
    data = await request.json()

    # Проверяем что камера существует
    camera = get_camera(camera_id)
    if not camera:
        return JSONResponse({'error': 'Камера не найдена'}, status_code=404)

    # Обновляем только переданные поля
    updates = {}
    if 'name' in data:
        updates['name'] = data['name']
    if 'url' in data:
        updates['url'] = data['url']
    if 'enabled' in data:
        updates['enabled'] = data['enabled']

    if updates:
        update_camera(camera_id, updates)

    return {'status': 'ok', 'camera_id': camera_id}


@app.delete("/api/cameras/{camera_id}")
async def delete_camera_api(camera_id: str):
    """Удалить камеру"""
    if delete_camera(camera_id):
        return {'status': 'ok'}
    else:
        return JSONResponse({'error': 'Нельзя удалить последнюю камеру'}, status_code=400)
