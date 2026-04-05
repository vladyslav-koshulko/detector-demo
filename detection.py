import time
import torch
import numpy as np
from rfdetr import RFDETRBase
import supervision as sv

# 1. Ініціалізація моделі (автоматично завантажить ваги DINOv2)
# Для RTX 4060 краще почати з Base
model = RFDETRBase()

# 2. Створюємо фейковий кадр (симуляція камери 640x640)
fake_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 3. Тест швидкості (Benchmark)
print("Запуск тесту RF-DETR на RTX 4060...")
iterations = 50
start_time = time.time()

# Вимикаємо градієнти для прискорення інференсу
with torch.no_grad():
    for _ in range(iterations):
        # Метод predict повертає об'єкт sv.Detections
        detections = model.predict(fake_frame, threshold=0.5)

end_time = time.time()
total_time = end_time - start_time
fps = iterations / total_time

print(f"\n--- Результати для RF-DETR ---")
print(f"Середній FPS: {fps:.2f}")
print(f"Час на один кадр: {(total_time/iterations)*1000:.2f} мс")
