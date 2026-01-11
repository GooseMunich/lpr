import cv2
import time
import threading
from typing import Callable, Optional
import config

class RTSPCapture:
    """Захват кадров с RTSP потока"""
    
    def __init__(self, url: str, callback: Callable):
        self.url = url
        self.callback = callback
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None
        
    def start(self):
        """Запуск захвата"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"Захват RTSP запущен: {self.url}")
        
    def stop(self):
        """Остановка захвата"""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.thread:
            self.thread.join(timeout=5)
        print("Захват RTSP остановлен")
        
    def _capture_loop(self):
        """Основной цикл захвата"""
        reconnect_delay = 5
        
        while self.running:
            try:
                self.cap = cv2.VideoCapture(self.url)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self.cap.isOpened():
                    print(f"Не удалось подключиться к {self.url}")
                    time.sleep(reconnect_delay)
                    continue
                
                print("Подключено к RTSP потоку")
                last_frame_time = 0
                
                while self.running and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        print("Потеряно соединение с камерой")
                        break
                    
                    current_time = time.time()
                    if current_time - last_frame_time >= config.FRAME_INTERVAL:
                        last_frame_time = current_time
                        try:
                            self.callback(frame)
                        except Exception as e:
                            print(f"Ошибка обработки кадра: {e}")
                            
            except Exception as e:
                print(f"Ошибка RTSP: {e}")
                
            finally:
                if self.cap:
                    self.cap.release()
                    
            if self.running:
                print(f"Переподключение через {reconnect_delay} сек...")
                time.sleep(reconnect_delay)
