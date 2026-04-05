import cv2
import torch
import numpy as np
import time
from rfdetr import RFDETRBase
from ultralytics import YOLO, RTDETR
import supervision as sv

# ================= НАЛАШТУВАННЯ =================
CAM_PER_MODEL = 1
scroll_offset = 0  # Поточна позиція прокрутки
zoom_level = 0.8   # Початковий масштаб
# ================================================

COCO_YOLO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COCO_RF = ['background'] + COCO_YOLO

class DetectionEngine:
    def __init__(self, name, model_obj, classes, threshold=0.5):
        self.name, self.model, self.classes, self.threshold = name, model_obj, classes, threshold
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

        for d in results:
            if hasattr(d, 'data'):
                keys = [k for k, v in d.data.items() if not isinstance(v, (np.ndarray, torch.Tensor, list))]
                for k in keys: del d.data[k]

        self.fps = len(frames) / (time.time() - t_start)
        return results

# Ініціалізація
engines = [
    DetectionEngine("RF-DETR", RFDETRBase(), COCO_RF, threshold=0.7),
    DetectionEngine("YOLOv8", YOLO("yolov8n.pt"), COCO_YOLO, threshold=0.3),
    DetectionEngine("RT-DETR", RTDETR("rtdetr-l.pt"), COCO_YOLO, threshold=0.5)
]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
cap = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
prev_gray = None

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_input = cv2.resize(frame, (480, 360))
        gray = cv2.cvtColor(frame_input, cv2.COLOR_BGR2GRAY)

        # 1. Розрахунок аналітики
        flow_view = np.zeros_like(frame_input)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(frame_input)
            hsv[..., 0], hsv[..., 1] = ang * 180 / np.pi / 2, 255
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_view = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        prev_gray = gray
        fg_view = cv2.cvtColor(bg_subtractor.apply(frame_input), cv2.COLOR_GRAY2BGR)
        red_view = frame_input.copy(); red_view[:, :, :2] = 0
        sob = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        sob_view = cv2.cvtColor(cv2.normalize(sob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        heat_view = cv2.addWeighted(frame_input, 0.5, cv2.applyColorMap(gray, cv2.COLORMAP_JET), 0.5, 0)

        columns = []
        for engine in engines:
            batch_results = engine.predict_batch([frame_input.copy()])
            for detections in batch_results:
                live = box_annotator.annotate(scene=frame_input.copy(), detections=detections)
                labels = [f"{engine.classes[int(id)]} {conf:.2f}" for id, conf in zip(detections.class_id, detections.confidence)]
                live = label_annotator.annotate(scene=live, detections=detections, labels=labels)
                cv2.putText(live, f"{engine.name} LIVE", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                stack = [live, flow_view, fg_view, red_view, sob_view, heat_view]
                titled_stack = []
                titles = ["LIVE", "OPTICAL FLOW", "FG MASK", "RED CHANNEL", "SOBEL", "HEATMAP"]
                for img, title in zip(stack, titles):
                    canvas = img.copy()
                    if title != "LIVE":
                        cv2.putText(canvas, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    titled_stack.append(canvas)

                columns.append(np.vstack(titled_stack))

        # 2. Збірка та Масштабування
        full_wall = np.hstack(columns)
        h, w = full_wall.shape[:2]

        # Застосування Zoom
        new_w, new_h = int(w * zoom_level), int(h * zoom_level)
        resized_wall = cv2.resize(full_wall, (new_w, new_h))

        # 3. Кроп для прокрутки (Scroll)
        display_h = 1000 # Висота вікна на екрані
        if new_h > display_h:
            # Обмежуємо зміщення
            scroll_offset = max(0, min(scroll_offset, new_h - display_h))
            view = resized_wall[scroll_offset : scroll_offset + display_h, :]
        else:
            view = resized_wall

        cv2.imshow("Smart Analysis Server", view)

        # 4. Обробка клавіш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('w'): scroll_offset -= 50 # Вгору
        elif key == ord('s'): scroll_offset += 50 # Вниз
        elif key == ord('=') or key == ord('+'): zoom_level += 0.1 # Zoom In
        elif key == ord('-'): zoom_level = max(0.2, zoom_level - 0.1) # Zoom Out
        elif key == ord('r'): zoom_level, scroll_offset = 0.8, 0 # Reset

finally:
    cap.release()
    cv2.destroyAllWindows()
