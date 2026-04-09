import cv2
import torch
import numpy as np
import os
import logging
import streamlit as st
import urllib.request
import urllib.parse
from ultralytics import YOLO, RTDETR
import supervision as sv

logger = logging.getLogger(__name__)
MODELS_DIR = "models"


def _ensure_models_dir():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

def get_available_models():
    models = ["Mock (No Model)", "Завантажити нову (YOLO)..."]
    if os.path.exists('.'):
        for f in os.listdir('.'):
            if f.endswith(('.pt', '.pth')):
                models.append(f)
    _ensure_models_dir()
    for f in os.listdir(MODELS_DIR):
        if f.endswith(('.pt', '.pth')):
            models.append(os.path.join(MODELS_DIR, f))
    return sorted(list(dict.fromkeys(models)))


def download_model_from_url(url):
    _ensure_models_dir()
    parsed = urllib.parse.urlparse(url)
    name = os.path.basename(parsed.path) or "downloaded_model.pt"
    safe_name = name.replace(" ", "_")
    target = os.path.join(MODELS_DIR, safe_name)
    urllib.request.urlretrieve(url, target)
    return target


def save_uploaded_model(file_bytes, filename):
    _ensure_models_dir()
    safe_name = os.path.basename(filename).replace(" ", "_")
    target = os.path.join(MODELS_DIR, safe_name)
    with open(target, "wb") as f:
        f.write(file_bytes)
    return target


def list_compute_devices():
    devices = [("auto", "Auto (GPU if available)"), ("cpu", "CPU")]
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            devices.append((f"cuda:{idx}", f"GPU{idx}: {name}"))
    return devices


def resolve_device(device_choice):
    if not device_choice or device_choice == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if isinstance(device_choice, str) and device_choice.startswith("cuda"):
        if torch.cuda.is_available():
            try:
                idx = int(device_choice.split(":")[1]) if ":" in device_choice else 0
                return device_choice if idx < torch.cuda.device_count() else "cpu"
            except (ValueError, IndexError):
                return "cpu"
        return "cpu"
    return "cpu"


def describe_device(device):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        try:
            idx = int(device.split(":")[1]) if ":" in device else 0
            if idx < torch.cuda.device_count():
                return f"GPU{idx}: {torch.cuda.get_device_name(idx)}"
        except (ValueError, IndexError):
            pass
    return "CPU"

class MockResult:
    def __init__(self, frame):
        self.orig_shape = frame.shape[:2]
        self.boxes = type('Boxes', (), {'cls': torch.tensor([]), 'conf': torch.tensor([]), 'xyxy': torch.tensor([]), 'data': torch.tensor([]), 'id': None})()
        self.names = {0: "mock_object"}

class MockModel:
    def __init__(self): self.names = {0: "mock_object"}
    def __call__(self, frames, **kwargs):
        if isinstance(frames, list): return [MockResult(f) for f in frames]
        return [MockResult(frames)]

@st.cache_resource
def load_detection_model(path, device=None):
    if path is None or "Mock" in str(path):
        return MockModel()
    if str(path).lower().endswith(".pth"):
        logger.warning("Unsupported model suffix for Ultralytics: %s (use .pt)", path)
        return MockModel()
    try:
        if "rtdetr" in path.lower():
            model = RTDETR(path)
        else:
            model = YOLO(path)
        if device:
            try:
                model.to(device)
            except Exception as exc:
                logger.warning("Failed to move model to device %s: %s", device, exc)
        return model
    except Exception as exc:
        logger.exception("Failed to load model %s: %s", path, exc)
        return MockModel()

class DetectionEngine:
    def __init__(self, model_path=None, threshold=0.5, device="auto"):
        self.threshold = threshold
        self.device_choice = device
        self.device = resolve_device(device)
        self.device_label = describe_device(self.device)
        self.model = load_detection_model(model_path, device=self.device)
        self.classes = getattr(self.model, 'names', {0: "object"})
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def get_all_classes(self): return list(self.classes.values())

    def process_frame(self, frame, view_type="Live", visibility_map=None, night_mode=False, quality="480p", zoom=1.0):
        if frame is None: return None, False
        
        if night_mode:
            lab = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
            lab[0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[0])
            frame = cv2.cvtColor(cv2.merge(lab), cv2.COLOR_LAB2BGR)
        
        if zoom > 1.0:
            h, w = frame.shape[:2]
            zh, zw = int(h / zoom), int(w / zoom)
            y, x = (h - zh) // 2, (w - zw) // 2
            frame = cv2.resize(frame[y:y+zh, x:x+zw], (w, h))

        h, w = frame.shape[:2]
        tw = 1280 if quality == "720p" else (854 if quality == "480p" else 640)
        work_frame = cv2.resize(frame, (tw, int(h * (tw / w))))
        
        if view_type == "Live":
            res = self.model(work_frame, conf=self.threshold, verbose=False, device=self.device)
            try:
                detections = sv.Detections.from_ultralytics(res[0])
                if visibility_map:
                    indices = [i for i, cid in enumerate(detections.class_id) if visibility_map.get(self.classes.get(int(cid)), True)]
                    detections = detections[indices]
                scene = self.box_annotator.annotate(scene=work_frame.copy(), detections=detections)
                labels = [f"{self.classes.get(int(id), 'obj')} {conf:.2f}" for id, conf in zip(detections.class_id, detections.confidence)]
                return self.label_annotator.annotate(scene=scene, detections=detections, labels=labels), len(detections) > 0
            except: return work_frame, False
        return work_frame, False
