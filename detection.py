import cv2
import torch
import time
import json
from rfdetr import RFDETRBase
from ultralytics import YOLO
import supervision as sv

# 1. Ініціалізація моделей
print("Завантаження моделей...")
model_rf = RFDETRBase()
model_yolo = YOLO("yolov8n.pt")

# Налаштування анотаторів
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Статистика
stats = {
    "rf": {"times": [], "fps_list": []},
    "yolo": {"times": [], "fps_list": []}
}
log_file = "fps_stats.log"
start_minute = time.time()

def save_stats(label, fps_data):
    if not fps_data: return
    avg_fps = sum(fps_data) / len(fps_data)
    min_fps = min(fps_data)
    max_fps = max(fps_data)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a") as f:
        line = f"[{timestamp}] Model: {label} | Avg: {avg_fps:.2f} | Min: {min_fps:.2f} | Max: {max_fps:.2f}\n"
        f.write(line)
    print(f"Статистику збережено для {label}")

# 2. Запуск камери
cap = cv2.VideoCapture(0)
print(f"Запис статистики активовано у файл: {log_file}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_time = time.time()

        # --- ТЕСТ RF-DETR ---
        t_start = time.time()
        with torch.no_grad():
            _ = model_rf.predict(frame, threshold=0.5)
        dt = time.time() - t_start
        fps_rf = 1 / dt if dt > 0 else 0
        stats["rf"]["fps_list"].append(fps_rf)

        # --- ТЕСТ YOLOv8 ---
        t_start = time.time()
        _ = model_yolo(frame, verbose=False)
        dt = time.time() - t_start
        fps_yolo = 1 / dt if dt > 0 else 0
        stats["yolo"]["fps_list"].append(fps_yolo)

        # --- ЗАПИС СТАТИСТИКИ ЩОХВИЛИНИ ---
        if current_time - start_minute >= 60:
            save_stats("RF-DETR", stats["rf"]["fps_list"])
            save_stats("YOLOv8", stats["yolo"]["fps_list"])
            # Очищення для наступної хвилини
            stats["rf"]["fps_list"] = []
            stats["yolo"]["fps_list"] = []
            start_minute = current_time

        # Відображення (опціонально, для контролю)
        cv2.putText(frame, f"RF FPS: {fps_rf:.1f} | YOLO FPS: {fps_yolo:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Benchmarks Running...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Записати останні дані перед виходом
    save_stats("RF-DETR (Final)", stats["rf"]["fps_list"])
    save_stats("YOLOv8 (Final)", stats["yolo"]["fps_list"])
    cap.release()
    cv2.destroyAllWindows()
