import cv2
import torch
import time
from rfdetr import RFDETRBase
from sympy import false
from ultralytics import YOLO
import supervision as sv

# 1. Ініціалізація моделей
print("Завантаження моделей...")
model_rf = RFDETRBase()
model_yolo = YOLO("yolov8n.pt")

# ПОРІГ ДЕТЕКЦІЇ (змінюємо тут)
RF_THRESHOLD = 0.8  # Піднімаємо, щоб прибрати помилки
YOLO_THRESHOLD = 0.3 # Залишаємо нижчим, бо YOLO обережніша

# Налаштування анотаторів
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Базовий список COCO (без фону, як у YOLO)
COCO_YOLO = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Список для RF-DETR (з фоном на початку)
COCO_RF = ['background'] + COCO_YOLO

# Файли логів
LOG_RF = "rf_detr_filtered.log"
LOG_YOLO = "yolo_filtered.log"

def log_detection(file_path, fps, detections, threshold, type=False):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    COCO_CLASSES = COCO_RF if type else COCO_YOLO
    found_objects = [
        f"{COCO_CLASSES[int(id)] if int(id) < len(COCO_CLASSES) else id}({conf:.2f})"
        for id, conf in zip(detections.class_id, detections.confidence)
    ]
    objs_str = ", ".join(found_objects) if found_objects else "None"
    with open(file_path, "a") as f:
        f.write(f"[{timestamp}] Thr: {threshold} | FPS: {fps:.2f} | Objects: {objs_str}\n")

def get_annotated_frame(frame, detections, label_prefix, fps, threshold, classes_list):
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    # Використовуємо переданий список класів
    labels = [
        f"{classes_list[int(id)] if int(id) < len(classes_list) else id} {conf:.2f}"
        for id, conf in zip(detections.class_id, detections.confidence)
    ]
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    cv2.putText(annotated, f"{label_prefix} FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return annotated

# 2. Запуск камери
cap = cv2.VideoCapture(0)
print(f"Запуск з порогом RF-DETR = {RF_THRESHOLD}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- ОБРОБКА RF-DETR (з новим порогом) ---
        t_start = time.time()
        with torch.no_grad():
            det_rf = model_rf.predict(frame, threshold=RF_THRESHOLD)
        fps_rf = 1 / (time.time() - t_start)
        res_rf = get_annotated_frame(frame, det_rf, "RF-DETR", fps_rf, RF_THRESHOLD, COCO_RF)
        log_detection(LOG_RF, fps_rf, det_rf, RF_THRESHOLD, type=True)

        # --- ОБРОБКА YOLOv8 ---
        t_start = time.time()
        results_yolo = model_yolo(frame, conf=YOLO_THRESHOLD, verbose=False)
        det_yolo = sv.Detections.from_ultralytics(results_yolo[0])
        fps_yolo = 1 / (time.time() - t_start)
        res_yolo = get_annotated_frame(frame, det_yolo, "YOLOv8", fps_yolo, YOLO_THRESHOLD, COCO_YOLO)
        log_detection(LOG_YOLO, fps_yolo, det_yolo, YOLO_THRESHOLD, type=False)

        # 3. Відображення
        cv2.imshow("RF-DETR (High Confidence)", res_rf)
        cv2.imshow("YOLOv8 (Standard)", res_yolo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
