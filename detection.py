import cv2
import torch
import time
from rfdetr import RFDETRBase
from ultralytics import YOLO
import supervision as sv

# 1. Ініціалізація моделей
print("Завантаження моделей...")
model_rf = RFDETRBase()
model_yolo = YOLO("yolov8n.pt") # Найлегша версія для швидкості

# 2. Налаштування анотаторів
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Список класів COCO (спільний для обох моделей)
COCO_CLASSES = [
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

def process_frame(frame, detections, model_name, fps):
    # Малюємо рамки
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)

    # Створюємо підписи
    labels = [
        f"{COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else class_id} {conf:.2f}"
        for class_id, conf in zip(detections.class_id, detections.confidence)
    ]

    # Малюємо текст
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # Вивід FPS
    cv2.putText(annotated, f"{model_name} FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return annotated

# 3. Запуск камери
cap = cv2.VideoCapture(0)
print("Запуск паралельної детекції. Натисніть 'q' для виходу.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- ТЕСТ RF-DETR ---
    t1 = time.time()
    with torch.no_grad():
        det_rf = model_rf.predict(frame, threshold=0.5)
    fps_rf = 1 / (time.time() - t1)
    res_rf = process_frame(frame, det_rf, "RF-DETR (Transformer)", fps_rf)

    # --- ТЕСТ YOLOv8 ---
    t2 = time.time()
    results_yolo = model_yolo(frame, verbose=False)[0]
    det_yolo = sv.Detections.from_ultralytics(results_yolo)
    fps_yolo = 1 / (time.time() - t2)
    res_yolo = process_frame(frame, det_yolo, "YOLOv8 (CNN)", fps_yolo)

    # 4. Відображення двох вікон
    cv2.imshow("RF-DETR", res_rf)
    cv2.imshow("YOLOv8", res_yolo)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
