import cv2
import threading
import time
import os
import logging
from collections import deque
from datetime import datetime

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

    def start(self, source_id):
        if source_id is None or source_id == "": return False
        if source_id in self.running and self.running[source_id]: return True
            
        try:
            # Check if it's a static image
            ext = os.path.splitext(str(source_id))[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                frame = cv2.imread(source_id)
                if frame is not None:
                    self.frames[source_id] = frame
                    self.is_image[source_id] = True
                    self.running[source_id] = True
                    logger.info(f"Static image loaded: {source_id}")
                    return True
                else:
                    logger.error(f"Failed to load image: {source_id}")
                    return False

            # Otherwise treat as video/camera
            source = int(source_id) if (isinstance(source_id, str) and source_id.isdigit()) else source_id
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            
            if cap.isOpened():
                self.caps[source_id] = cap
                self.is_image[source_id] = False
                self.running[source_id] = True
                self.fps_counters[source_id] = deque(maxlen=30)
                thread = threading.Thread(target=self._update, args=(source_id,), daemon=True)
                self.threads[source_id] = thread
                thread.start()
                return True
        except Exception as e:
            logger.exception(f"Error starting {source_id}: {e}")
        return False

    def _update(self, source_id):
        prev_time = time.time()
        while self.running.get(source_id, False):
            ret, frame = self.caps[source_id].read()
            if not ret:
                # Loop for files
                if isinstance(source_id, str) and ('.' in source_id or 'rtsp' in source_id):
                    self.caps[source_id].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.running[source_id] = False
                break
            
            curr_time = time.time()
            dt = curr_time - prev_time
            if dt > 0:
                self.fps_counters[source_id].append(1.0 / dt)
            prev_time = curr_time
            
            self.frames[source_id] = frame
            if self.recording_flags.get(source_id, False):
                self._write_to_record(source_id, frame)
            time.sleep(0.001)

    def _write_to_record(self, source_id, frame):
        if source_id not in self.recorders:
            if not os.path.exists("recordings"): os.makedirs("recordings")
            h, w = frame.shape[:2]
            filename = f"recordings/rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.recorders[source_id] = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
        self.recorders[source_id].write(frame)

    def get_fps(self, source_id):
        if self.is_image.get(source_id, False): return 0
        q = self.fps_counters.get(source_id, [])
        return sum(q) / len(q) if q else 0

    def get_frame(self, source_id):
        return self.frames.get(source_id, None)

    def stop(self, source_id):
        self.stop_recording(source_id)
        self.running[source_id] = False
        if source_id in self.caps:
            self.caps[source_id].release()
            del self.caps[source_id]
        if source_id in self.frames: del self.frames[source_id]

    def stop_all(self):
        for s_id in list(self.caps.keys()): self.stop(s_id)

    def start_recording(self, source_id): self.recording_flags[source_id] = True
    def stop_recording(self, source_id):
        self.recording_flags[source_id] = False
        if source_id in self.recorders:
            self.recorders[source_id].release()
            del self.recorders[source_id]

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
