import cv2
import torch
from rfdetr import RFDETRBase
import supervision as sv

# 1. Ініціалізація моделі
print("Завантаження моделі RF-DETR...")
model = RFDETRBase()

# Спробуємо дістати класи автоматично або використаємо стандартний список COCO
try:
    # Деякі версії зберігають класи в model.model.classes або model.classes
    if hasattr(model, 'classes'):
        CLASS_NAMES = model.classes
    elif hasattr(model.model, 'classes'):
        CLASS_NAMES = model.model.classes
    else:
        # Стандартний список COCO (80 класів), якщо нічого не знайдено
        CLASS_NAMES = [
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
except Exception:
    CLASS_NAMES = []

# 2. Налаштування анотаторів (оновлені класи в Supervision)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 3. Підключення до камери
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Помилка: Не вдалося відкрити камеру.")
    exit()

print("Детекція запущена. Натисніть 'q' для виходу.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Детекція (без оптимізації, щоб уникнути TracerWarning поки що)
    with torch.no_grad():
        detections = model.predict(frame, threshold=0.5)

    # 5. Візуалізація
    labels = [
        f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else class_id} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Малюємо рамки
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # Малюємо текст
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Показуємо результат
    cv2.imshow("RF-DETR Demo", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
