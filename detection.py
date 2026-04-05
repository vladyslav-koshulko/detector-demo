import cv2
import torch
import numpy as np
import time
from rfdetr import RFDETRBase
from ultralytics import YOLO, RTDETR
import supervision as sv

# ================= НАЛАШТУВАННЯ =================
CAM_PER_MODEL = 2# Скільки камер на кожну модель
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
            # RF-DETR може повертати один об'єкт або список
            raw_res = self.model.predict(frames, threshold=self.threshold)
            results = raw_res if isinstance(raw_res, list) else [raw_res]
        elif isinstance(self.model, (YOLO, RTDETR)):
            res = self.model(frames, conf=self.threshold, verbose=False)
            results = [sv.Detections.from_ultralytics(r) for r in res]

        # ПРИМУСОВЕ ОЧИЩЕННЯ ТА ПЕРЕВІРКА
        final_results = []
        for d in results:
            # Якщо d - це кортеж (що буває в RF-DETR), беремо перший елемент
            det_obj = d[0] if isinstance(d, tuple) else d

            if hasattr(det_obj, 'data'):
                # Видаляємо все, що не є масивом (source_shape тощо)
                bad_keys = [k for k, v in det_obj.data.items()
                            if not isinstance(v, (np.ndarray, torch.Tensor, list))]
                for k in bad_keys:
                    del det_obj.data[k]
            final_results.append(det_obj)

        dt = time.time() - t_start
        self.fps = len(frames) / dt if dt > 0 else 0
        return final_results

print("Завантаження моделей...")
engines = [
    DetectionEngine("RF-DETR", RFDETRBase(), COCO_RF, threshold=0.4),
    DetectionEngine("YOLOv8", YOLO("yolov8n.pt"), COCO_YOLO, threshold=0.4),
    DetectionEngine("RT-DETR", RTDETR("rtdetr-l.pt"), COCO_YOLO, threshold=0.4),
    DetectionEngine("YOLOv10", YOLO("yolov10n.pt"), COCO_YOLO, threshold=0.4)

]

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

def create_grid(frames):
    if not frames:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    n = len(frames)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # Додаємо порожні кадри для заповнення сітки
    sample_frame = frames[0]
    while len(frames) < rows * cols:
        frames.append(np.zeros_like(sample_frame))

    grid_rows = [np.hstack(frames[i*cols : (i+1)*cols]) for i in range(rows)]
    return np.vstack(grid_rows)

print(f"Сервер запущено: {len(engines)} моделей. Всього {len(engines)*CAM_PER_MODEL} вікон.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_input = cv2.resize(frame, (640, 480))
        all_annotated_frames = []

        for engine in engines:
            batch = [frame_input.copy() for _ in range(CAM_PER_MODEL)]
            batch_results = engine.predict_batch(batch)

            for i, detections in enumerate(batch_results):
                # Тепер detections — це точно чистий sv.Detections
                annotated = box_annotator.annotate(scene=batch[i], detections=detections)

                labels = [
                    f"{engine.classes[int(id)] if int(id) < len(engine.classes) else id} {conf:.2f}"
                    for id, conf in zip(detections.class_id, detections.confidence)
                ]

                annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

                cv2.putText(annotated, f"{engine.name} Cam {i+1}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated, f"FPS: {engine.fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                all_annotated_frames.append(annotated)


        final_grid = create_grid(all_annotated_frames)

        # Масштабування для виводу
        h, w = final_grid.shape[:2]
        target_w = 1280
        scale = target_w / w
        display_res = cv2.resize(final_grid, (0, 0), fx=scale, fy=scale)

        cv2.imshow("Multi-Model Detection Server", display_res)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
