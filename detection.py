import cv2
import torch
import numpy as np
import time
from rfdetr import RFDETRBase
from ultralytics import YOLO, RTDETR
import supervision as sv

# ================= НАЛАШТУВАННЯ =================
CAM_PER_MODEL = 1        # Камер на модель
SHOW_TECHNICAL_VIEW = True  # Режим "Технічний зір" (Gray/Edges)
SHOW_ATTENTION_MAP = True   # Карта уваги (Heatmap) для RF-DETR (потребує GPU!)
# ================================================

COCO_YOLO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COCO_RF = ['background'] + COCO_YOLO

class DetectionEngine:
    def __init__(self, name, model_obj, classes, threshold=0.5):
        self.name = name
        self.model = model_obj
        self.classes = classes
        self.threshold = threshold
        self.fps = 0

    def predict_batch(self, frames):
        t_start = time.time()
        results = []
        if isinstance(self.model, RFDETRBase):
            raw_res = self.model.predict(frames, threshold=self.threshold)
            results = raw_res if isinstance(raw_res, list) else [raw_res]
        elif isinstance(self.model, (YOLO, RTDETR)):
            res = self.model(frames, conf=self.threshold, verbose=False)
            results = [sv.Detections.from_ultralytics(r) for r in res]

        for i in range(len(results)):
            d = results[i]
            if hasattr(d, 'data'):
                keys_to_del = [k for k, v in d.data.items() if not isinstance(v, (np.ndarray, torch.Tensor, list))]
                for k in keys_to_del: del d.data[k]

        dt = time.time() - t_start
        self.fps = len(frames) / dt if dt > 0 else 0
        return results

# 1. Ініціалізація
print("Завантаження моделей...")
engines = [
    DetectionEngine("RF-DETR", RFDETRBase(), COCO_RF, threshold=0.7),
    DetectionEngine("YOLOv8", YOLO("yolov8n.pt"), COCO_YOLO, threshold=0.3),
    DetectionEngine("RT-DETR", RTDETR("rtdetr-l.pt"), COCO_YOLO, threshold=0.5),
    DetectionEngine("YOLOv10", YOLO("yolov10n.pt"), COCO_YOLO, threshold=0.4)
]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
cap = cv2.VideoCapture(0)

def generate_heatmap(frame):
    """Симуляція карти уваги на основі градієнтів яскравості (Fast Pseudo-Attention)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

def create_grid(frames):
    if not frames: return np.zeros((480, 640, 3), dtype=np.uint8)
    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    while len(frames) < rows * cols: frames.append(np.zeros_like(frames[0]))
    grid_rows = [np.hstack(frames[i*cols : (i+1)*cols]) for i in range(rows)]
    return np.vstack(grid_rows)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_input = cv2.resize(frame, (640, 480))
        all_annotated_frames = []

        # Основні моделі
        for engine in engines:
            batch = [frame_input.copy() for _ in range(CAM_PER_MODEL)]
            batch_results = engine.predict_batch(batch)
            for i, detections in enumerate(batch_results):
                annotated = box_annotator.annotate(scene=batch[i], detections=detections)
                labels = [f"{engine.classes[int(id)] if int(id) < len(engine.classes) else id} {conf:.2f}"
                          for id, conf in zip(detections.class_id, detections.confidence)]
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                cv2.putText(annotated, f"{engine.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                all_annotated_frames.append(annotated)

        # ДОДАТКОВІ КАМЕРИ ("Очі" моделі)
        if SHOW_TECHNICAL_VIEW:
            tech_view = cv2.Canny(frame_input, 100, 200) # Краї (Edges)
            tech_view = cv2.cvtColor(tech_view, cv2.COLOR_GRAY2BGR)
            cv2.putText(tech_view, "TECH: Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            all_annotated_frames.append(tech_view)

        if SHOW_ATTENTION_MAP:
            att_view = generate_heatmap(frame_input) # Карта "уваги"
            cv2.putText(att_view, "ATTENTION: Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            all_annotated_frames.append(att_view)

        final_grid = create_grid(all_annotated_frames)
        h, w = final_grid.shape[:2]
        scale = 1280 / w
        display_res = cv2.resize(final_grid, (0, 0), fx=scale, fy=scale)
        cv2.imshow("Detection Server: Diagnostic View", display_res)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
