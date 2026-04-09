# 🕵️ Vision Intelligence System (Hybrid Web/Desktop)

Професійна гібридна система відеоаналітики та комп'ютерного зору. Проєкт поєднує зручність веб-інтерфейсу (Streamlit) для віддаленого моніторингу та потужність десктопного рішення (OpenCV Native) для локальної роботи з мінімальними затримками.

## 🌟 Основні можливості

*   **Hybrid Interface:** Одночасне відображення потоків у браузері (Web) та у нативних вікнах OpenCV (Desktop Mode).
*   **Multi-Source Matrix:** Підтримка до 4 незалежних вікон аналітики. Кожне вікно має власні налаштування:
    *   **Джерела:** Вебкамери, IP-камери (RTSP/HTTP), завантажені відеофайли.
    *   **Моделі:** YOLOv8, RT-DETR, YOLOv10 або Mock-режим (для демонстрації без GPU).
    *   **Режими:** Live Detection, Optical Flow, FG Mask (рух), Sobel (контури), Heatmap.
*   **Batch Folder Detection:** Обробка папок з відео/зображеннями (локальні або мережеві) з експортом ZIP.
*   **Web Model Manager:** Завантаження моделей через URL або локальний файл у Web UI.
*   **Real-time Web Controls:** Вибір джерела/моделі та запуск/зупинка потоків напряму у браузері.
*   **Per-slot CPU/GPU:** Вибір пристрою для кожного слота з відображенням активного GPU/CPU.
*   **Platform Streams:** Підтримка YouTube/платформ через yt-dlp (кеш resolved URL).
*   **Refresh URL:** Оновлення cached URL для платформних стрімів у web/desktop.
*   **Web Multi-Stream View:** Вибір N слотів для одночасного перегляду у браузері.
*   **Theme Toggle:** Світла/темна тема у web та desktop.
*   **Resizable Sidebar:** Сайдбар desktop можна масштабувати/згортати.
*   **Slot Init:** Desktop стартує з 1 слотом, інші додаються вручну.
*   **Individual & Global Control:** Кожне вікно можна запускати, зупиняти та записувати окремо. Також є глобальні кнопки "Запустити все" та "Стоп все".
*   **Advanced Recording:** Можливість запису відеопотоку у форматі MP4 безпосередньо з інтерфейсу.
*   **Performance Monitoring:** Відображення реального FPS для кожного джерела окремо.
*   **GPU Acceleration:** Автоматичне виявлення та використання NVIDIA CUDA для прискорення нейромереж.

## 🛠️ Технологічний стек

*   **Python 3.10+**
*   **Streamlit** (Web UI)
*   **OpenCV** (Video Stream, Desktop UI, Computer Vision filters)
*   **Ultralytics** (YOLO, RT-DETR inference)
*   **Supervision** (Annotations & Data handling)
*   **PyTorch** (Deep Learning backend)

## 📦 Встановлення та запуск

### 1. Підготовка середовища

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 2. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 3. Запуск

```bash
streamlit run app.py
```

## 🎮 Керування системою

1.  **Бічна панель (Sidebar):** Налаштуйте джерела та моделі для кожного вікна.
2.  **Запуск:** Використовуйте кнопку `▶️ Start X` для запуску конкретного вікна або `🚀 ЗАПУСК УСІХ` для всіх одразу.
3.  **Запис:** Натисніть `🔴 Rec`, щоб розпочати запис. Файли зберігаються у папку `recordings/`.
4.  **Desktop View:** Вікна OpenCV відкриваються автоматично при запуску потоку. Для закриття натисніть `Q` у вікні OpenCV або кнопку `Stop` у веб-інтерфейсі.

---
*Розроблено як Hybrid Architecture Pet-project.*
