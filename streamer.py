import cv2
import threading
import time
import os
import logging
from collections import deque
from datetime import datetime

try:
    from yt_dlp import YoutubeDL
except Exception:  # Optional dependency
    YoutubeDL = None

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiCameraStreamer:
    def __init__(self):
        self.caps = {}
        self.frames = {}
        self.running = {}
        self.threads = {}
        self.fps_counters = {}
        self.recorders = {}
        self.recording_flags = {}
        self.is_image = {} # Track if source is a static image
        self.resolved_urls = {}
        self.stream_sources = {}
        self.cache_path = "stream_cache.json"
        self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return
        try:
            import json
            with open(self.cache_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.resolved_urls.update(data)
        except Exception as exc:
            logger.warning("Failed to load stream cache: %s", exc)

    def _save_cache(self):
        try:
            import json
            with open(self.cache_path, "w") as f:
                json.dump(self.resolved_urls, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to save stream cache: %s", exc)

    def _resolve_stream_url(self, source_id, force=False):
        if not isinstance(source_id, str):
            return source_id
        if not force and source_id in self.resolved_urls:
            return self.resolved_urls[source_id]
        if not source_id.startswith("http"):
            return source_id
        if YoutubeDL is None:
            logger.warning("yt-dlp is not installed. Using raw URL: %s", source_id)
            return source_id
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
            "noplaylist": True,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_id, download=False)
                resolved = info.get("url") or source_id
                self.resolved_urls[source_id] = resolved
                self._save_cache()
                logger.info("Resolved stream URL via yt-dlp: %s", source_id)
                return resolved
        except Exception as exc:
            logger.warning("Failed to resolve URL %s: %s", source_id, exc)
            return source_id

    def refresh_stream_url(self, source_id, restart=False):
        if source_id in self.resolved_urls:
            del self.resolved_urls[source_id]
            self._save_cache()
        resolved = self._resolve_stream_url(source_id, force=True)
        if restart:
            keys_to_restart = [
                key for key, src in self.stream_sources.items()
                if src == source_id and self.running.get(key)
            ]
            for key in keys_to_restart:
                self.stop(key)
                self.start(source_id, stream_key=key)
        return resolved

    def start(self, source_id, stream_key=None, refresh=False):
        if source_id is None or source_id == "": return False
        stream_key = str(source_id) if stream_key is None else stream_key
        if stream_key in self.running and self.running[stream_key]: return True
            
        try:
            # Check if it's a static image
            ext = os.path.splitext(str(source_id))[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                frame = cv2.imread(source_id)
                if frame is not None:
                    self.frames[stream_key] = frame
                    self.is_image[stream_key] = True
                    self.running[stream_key] = True
                    self.stream_sources[stream_key] = source_id
                    logger.info(f"Static image loaded: {source_id}")
                    return True
                else:
                    logger.error(f"Failed to load image: {source_id}")
                    return False

            # Otherwise treat as video/camera/stream
            resolved = self._resolve_stream_url(source_id, force=refresh)
            source = int(resolved) if (isinstance(resolved, str) and resolved.isdigit()) else resolved
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            if cap.isOpened():
                self.caps[stream_key] = cap
                self.is_image[stream_key] = False
                self.running[stream_key] = True
                self.stream_sources[stream_key] = source_id
                self.fps_counters[stream_key] = deque(maxlen=30)
                thread = threading.Thread(target=self._update, args=(stream_key,), daemon=True)
                self.threads[stream_key] = thread
                thread.start()
                return True
        except Exception as e:
            logger.exception(f"Error starting {source_id}: {e}")
        return False

    def _update(self, stream_key):
        prev_time = time.time()
        while self.running.get(stream_key, False):
            ret, frame = self.caps[stream_key].read()
            if not ret:
                # Loop for files
                source_id = self.stream_sources.get(stream_key, stream_key)
                if isinstance(source_id, str) and ('.' in source_id or 'rtsp' in source_id):
                    self.caps[stream_key].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.running[stream_key] = False
                break
            
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                self.fps_counters[stream_key].append(1.0 / dt)
            prev_time = curr_time
            
            self.frames[stream_key] = frame
            if self.recording_flags.get(stream_key, False):
                self._write_to_record(stream_key, frame)
            time.sleep(0.001)

    def _write_to_record(self, source_id, frame):
        if source_id not in self.recorders:
            if not os.path.exists("recordings"): os.makedirs("recordings")
            h, w = frame.shape[:2]
            filename = f"recordings/rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recorders[source_id] = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        self.recorders[source_id].write(frame)

    def get_fps(self, stream_key):
        if self.is_image.get(stream_key, False): return 0
        q = self.fps_counters.get(stream_key, [])
        return sum(q) / len(q) if q else 0

    def get_frame(self, stream_key):
        return self.frames.get(stream_key, None)

    def stop(self, stream_key):
        self.stop_recording(stream_key)
        self.running[stream_key] = False
        if stream_key in self.caps:
            self.caps[stream_key].release()
            del self.caps[stream_key]
        if stream_key in self.frames:
            del self.frames[stream_key]
        if stream_key in self.stream_sources:
            del self.stream_sources[stream_key]

    def stop_all(self):
        for s_id in list(self.caps.keys()): self.stop(s_id)

    def start_recording(self, stream_key): self.recording_flags[stream_key] = True
    def stop_recording(self, stream_key):
        self.recording_flags[stream_key] = False
        if stream_key in self.recorders:
            self.recorders[stream_key].release()
            del self.recorders[stream_key]

    def save_snapshot(self, frame, prefix="result"):
        if not os.path.exists("results"): os.makedirs("results")
        filename = f"results/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        return filename

def scan_cameras(limit=5):
    available = []
    for i in range(limit):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(str(i))
            cap.release()
    return available
