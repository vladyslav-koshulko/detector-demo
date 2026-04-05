import cv2
import torch
import numpy as np
import time
from rfdetr import RFDETRBase
from ultralytics import YOLO, RTDETR
import supervision as sv

# ================= НАЛАШТУВАННЯ =================
CAM_PER_MODEL = 1
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

# Глобальні змінні для аналітики
prev_gray = None
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def apply_analytics(frame, prev_gray):
    # 1. Optical Flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow_view = np.zeros_like(frame)
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 0], hsv[..., 1] = ang * 180 / np.pi / 2, 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_view = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. Foreground Mask
    fg_mask = bg_subtractor.apply(frame)
    fg_view = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    # 3. Color Deconvolution (RED channel focus)
    red_view = frame.copy()
    red_view[:, :, 0] = 0 # Blue
    red_view[:, :, 1] = 0 # Green

    # 4. Sobel Filter (High-Pass)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_view = cv2.cvtColor(cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # 5. Heatmap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heat_view = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)

    return flow_view, fg_view, red_view, sobel_view, heat_view, gray

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_input = cv2.resize(frame, (320, 240)) # Зменшуємо розмір, бо вікон дуже багато

        # Розраховуємо аналітику один раз
        flow, fg, red, sob, heat, prev_gray = apply_analytics(frame_input, prev_gray)

        columns = []
        for engine in engines:
            batch = [frame_input.copy()]
            batch_results = engine.predict_batch(batch)

            for i, detections in enumerate(batch_results):
                # Рівень 1: LIVE (з баундбоксами)
                live = box_annotator.annotate(scene=batch[i].copy(), detections=detections)
                labels = [f"{engine.classes[int(id)] if int(id) < len(engine.classes) else id} {conf:.2f}" for id, conf in zip(detections.class_id, detections.confidence)]
                live = label_annotator.annotate(scene=live, detections=detections, labels=labels)
                cv2.putText(live, f"{engine.name} LIVE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Рівні 2-6: DEBUG (без баундбоксів)
                views = [
                    (flow, "OPTICAL FLOW"), (fg, "FG MASK"), (red, "COLOR: RED"), (sob, "SOBEL/HIGH-PASS"), (heat, "HEATMAP")
                ]

                stack = [live]
                for v, title in views:
                    tmp = v.copy()
                    cv2.putText(tmp, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    stack.append(tmp)

                columns.append(np.vstack(stack))

        final_wall = np.hstack(columns)
        # Масштабуємо, щоб влізло по висоті (наприклад, 1000 пікселів)
        h_total = final_wall.shape[0]
        scale = 1000 / h_total
        cv2.imshow("Super-Resolution Neural Analysis", cv2.resize(final_wall, (0, 0), fx=scale, fy=scale))

        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
