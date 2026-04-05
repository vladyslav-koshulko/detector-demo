import cv2
import torch
import numpy as np
import os
import logging
import streamlit as st
from ultralytics import YOLO, RTDETR
import supervision as sv

logger = logging.getLogger(__name__)

class MockResult:
    def __init__(self, frame):
        self.orig_shape = frame.shape[:2]
        self.boxes = type('Boxes', (), {'cls': torch.tensor([]), 'conf': torch.tensor([]), 'xyxy': torch.tensor([]), 'data': torch.tensor([])})()
        self.masks = None
        self.probs = None
        self.keypoints = None
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'postprocess': 0.0}
        self.names = {0: "mock_object"}

class MockModel:
    def __init__(self):
        self.names = {0: "mock_object"}
    def __call__(self, frames, **kwargs):
        if isinstance(frames, list):
            return [MockResult(f) for f in frames]
        return [MockResult(frames)]

@st.cache_resource
def load_detection_model(path):
    if path is None or "Mock" in str(path):
        return MockModel()
    
    logger.info(f"Loading model from {path}...")
    try:
        ext = os.path.splitext(path)[1].lower()
        if "rtdetr" in path.lower():
            model = RTDETR(path)
        elif ext == ".pt":
            model = YOLO(path)
        else:
            model = MockModel()
        return model
    except Exception as e:
        logger.error(f"Failed to load model {path}: {e}")
        return MockModel()

class DetectionEngine:
    def __init__(self, model_path=None, threshold=0.5):
        self.threshold = threshold
        self.model = load_detection_model(model_path)
        self.classes = getattr(self.model, 'names', {0: "object"})
        
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.prev_gray = None
        self.stats = []

    def get_all_classes(self):
        return list(self.classes.values())

    def process_frame(self, frame, view_type="Live", visibility_map=None, night_mode=False, quality="480p"):
        if frame is None: return None, False
        
        # 🌙 Night Mode Pre-processing (CLAHE)
        if night_mode:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Performance Resize
        h, w = frame.shape[:2]
        target_w = 640 if quality == "Native" else (1280 if quality == "720p" else 640)
        target_h = int(h * (target_w / w))
        work_frame = cv2.resize(frame, (target_w, target_h))
        
        found_trigger = False
        
        if view_type == "Live":
            results = self.model(work_frame, conf=self.threshold, verbose=False)
            try:
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Dynamic Visibility Filter
                if visibility_map:
                    mask = np.array([visibility_map.get(self.classes.get(int(cid)), True) for cid in detections.class_id])
                    if mask.size > 0:
                        detections = detections[mask]
                    else:
                        detections = sv.Detections.empty()
                
                found_trigger = len(detections) > 0
            except:
                detections = sv.Detections.empty()
            
            annotated_frame = self.box_annotator.annotate(scene=work_frame.copy(), detections=detections)
            labels = [f"{self.classes.get(int(id), 'obj')} {conf:.2f}" 
                     for id, conf in zip(detections.class_id, detections.confidence)]
            return self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels), found_trigger

        elif view_type == "Heatmap":
            gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
            return cv2.addWeighted(work_frame, 0.5, cv2.applyColorMap(gray, cv2.COLORMAP_JET), 0.5, 0), False

        return work_frame, False

def get_available_models():
    models = ["Mock (No Model)"]
    if os.path.exists('.'):
        for f in os.listdir('.'):
            if f.endswith(('.pt', '.pth')):
                models.append(f)
    return models
