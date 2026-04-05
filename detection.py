import cv2
import torch
import numpy as np
import time
from rfdetr import RFDETRBase
from ultralytics import YOLO, RTDETR
import supervision as sv

# ================= НАЛАШТУВАННЯ =================
CAM_PER_MODEL = 1        # Кількість основних камер на модель
SHOW_TRIPLE_STACK = True # Ввімкнути 3-рівневий аналіз під кожною моделлю
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

        self.fps = len(frames) / (time.time() - t_start)
        return results

# Ініціалізація
engines = [
    DetectionEngine("RF-DETR", RFDETRBase(), COCO_RF, threshold=0.7),
    DetectionEngine("YOLOv8", YOLO("yolov8n.pt"), COCO_YOLO, threshold=0.3),
    DetectionEngine("RT-DETR", RTDETR("rtdetr-l.pt"), COCO_YOLO, threshold=0.5),
    DetectionEngine("YOLOv10", YOLO("yolov10n.pt"), COCO_YOLO, threshold=0.4)
]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
cap = cv2.VideoCapture(0)

def get_edges(frame):
    edges = cv2.Canny(frame, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def get_heatmap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Базовий розмір кожного "віконця"
        frame_input = cv2.resize(frame, (480, 360))
        columns = []

        for engine in engines:
            batch = [frame_input.copy() for _ in range(CAM_PER_MODEL)]
            batch_results = engine.predict_batch(batch)

            for i, detections in enumerate(batch_results):
                # 1. TOP: Основна детекція
                annotated = box_annotator.annotate(scene=batch[i], detections=detections)
                labels = [f"{engine.classes[int(id)] if int(id) < len(engine.classes) else id} {conf:.2f}"
                          for id, conf in zip(detections.class_id, detections.confidence)]
                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
                cv2.putText(annotated, f"{engine.name} LIVE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if SHOW_TRIPLE_STACK:
                    # 2. MIDDLE: Технічний вид (Edges)
                    tech_edges = get_edges(batch[i])
                    cv2.putText(tech_edges, "TECH: STRUCTURAL EDGES", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # 3. BOTTOM: Теплова карта (Heatmap)
                    tech_heat = get_heatmap(batch[i])
                    cv2.putText(tech_heat, "TECH: ATTENTION HEATMAP", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    # Збираємо стовпчик з 3-х кадрів
                    column = np.vstack((annotated, tech_edges, tech_heat))
                else:
                    column = annotated

                columns.append(column)

        # Склеюємо всі колонки моделей в один широкий ряд
        final_wall = np.hstack(columns)

        # Масштабуємо під монітор, щоб все влізло по висоті
        screen_h = 900
        h, w = final_wall.shape[:2]
        scale = screen_h / h
        display_res = cv2.resize(final_wall, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Multi-Model Deep Analysis Server", display_res)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
