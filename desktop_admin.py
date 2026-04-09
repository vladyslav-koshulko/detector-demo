import sys
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"] = "" 

import cv2
import threading
import subprocess
import time
import tempfile
import zipfile
import shutil
from datetime import datetime
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from engine import DetectionEngine, get_available_models, list_compute_devices
from streamer import MultiCameraStreamer, scan_cameras
from state_manager import StateManager

class VideoWidget(QtWidgets.QLabel):
    def __init__(self, slot_id, parent=None):
        super().__init__(parent)
        self.slot_id = slot_id
        self.setMinimumSize(400, 225)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px;")
        self.setText(f"Камера #{slot_id + 1}")
        self.setScaledContents(False)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.last_frame = None

    def update_frame(self, frame):
        if frame is None: return
        self.last_frame = frame.copy()
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self)
        save_act = menu.addAction("📸 Зберегти Snapshot")
        menu.addSeparator()
        z_in = menu.addAction("Zoom In")
        z_out = menu.addAction("Zoom Out")
        z_res = menu.addAction("Reset Zoom")
        
        act = menu.exec(self.mapToGlobal(pos))
        state = StateManager()
        cfg = next(s for s in state.data["slot_configs"] if s["id"] == self.slot_id)
        
        if act == save_act and self.last_frame is not None:
            if not os.path.exists("results"): os.makedirs("results")
            path = f"results/snapshot_{int(time.time())}.jpg"
            cv2.imwrite(path, cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR))
            QtWidgets.QMessageBox.information(self, "OK", f"Збережено: {path}")
        elif act == z_in: state.update_slot(self.slot_id, {"zoom": cfg["zoom"] * 1.2})
        elif act == z_out: state.update_slot(self.slot_id, {"zoom": max(1.0, cfg["zoom"] * 0.8)})
        elif act == z_res: state.update_slot(self.slot_id, {"zoom": 1.0})

class VisionAdminApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Intelligence v6 Pro")
        self.resize(1600, 950)
        self.state = StateManager()
        self.streamer = MultiCameraStreamer()
        self.engines = {}
        self.slot_device_choices = {}
        self.slot_device_labels = {}
        self.web_process = None
        self._init_ui()
        self.timer = QTimer(); self.timer.timeout.connect(self._update_loop); self.timer.start(30)

    def _init_ui(self):
        main = QtWidgets.QWidget(); self.setCentralWidget(main); layout = QtWidgets.QHBoxLayout(main)
        
        # Sidebar
        scroll = QtWidgets.QScrollArea(); scroll.setFixedWidth(420); scroll.setWidgetResizable(True); layout.addWidget(scroll)
        ctrl = QtWidgets.QWidget(); scroll.setWidget(ctrl); self.sidebar = QtWidgets.QVBoxLayout(ctrl)

        # Settings
        sg = QtWidgets.QGroupBox("🌐 System Control"); sl = QtWidgets.QVBoxLayout(sg)
        self.btn_web = QtWidgets.QPushButton("🚀 Start Web Server"); self.btn_web.clicked.connect(self.toggle_web); sl.addWidget(self.btn_web)
        
        gh = QtWidgets.QHBoxLayout(); gh.addWidget(QtWidgets.QLabel("Grid:")); self.spin = QtWidgets.QSpinBox(); self.spin.setRange(1,4); self.spin.setValue(self.state.data["grid_columns"]); self.spin.valueChanged.connect(self._grid_change); gh.addWidget(self.spin); sl.addLayout(gh)
        self.sidebar.addWidget(sg)

        # Users Table
        self.user_table = QtWidgets.QTableWidget(0, 3); self.user_table.setHorizontalHeaderLabels(["User", "Role", "Actions"]); self.sidebar.addWidget(self.user_table)

        # Sources Manager
        self.sidebar.addWidget(QtWidgets.QLabel("📂 Джерела (Saved Sources):"))
        self.src_list = QtWidgets.QListWidget(); self.sidebar.addWidget(self.src_list)
        bh = QtWidgets.QHBoxLayout()
        b_url = QtWidgets.QPushButton("+ URL"); b_url.clicked.connect(self._add_url)
        b_file = QtWidgets.QPushButton("+ File"); b_file.clicked.connect(self._add_file)
        b_folder = QtWidgets.QPushButton("+ Folder"); b_folder.clicked.connect(self._add_folder)
        bh.addWidget(b_url); bh.addWidget(b_file); bh.addWidget(b_folder); self.sidebar.addLayout(bh)

        # Folder Batch Processing
        fb = QtWidgets.QGroupBox("🗂️ Batch Folder Detection")
        fbl = QtWidgets.QVBoxLayout(fb)
        self.folder_path_label = QtWidgets.QLabel("Папка не вибрана")
        fbl.addWidget(self.folder_path_label)
        fb_buttons = QtWidgets.QHBoxLayout()
        self.btn_pick_folder = QtWidgets.QPushButton("📁 Обрати папку")
        self.btn_pick_folder.clicked.connect(self._select_batch_folder)
        fb_buttons.addWidget(self.btn_pick_folder)
        fbl.addLayout(fb_buttons)
        self.folder_model_cb = QtWidgets.QComboBox(); self.folder_model_cb.addItems(get_available_models()); fbl.addWidget(self.folder_model_cb)
        self.btn_run_batch = QtWidgets.QPushButton("▶️ Запустити та зберегти ZIP")
        self.btn_run_batch.clicked.connect(self._run_batch_folder)
        fbl.addWidget(self.btn_run_batch)
        self.sidebar.addWidget(fb)

        # Config Area
        self.slot_area = QtWidgets.QVBoxLayout(); self.sidebar.addLayout(self.slot_area)
        self.btn_add = QtWidgets.QPushButton("➕ Add New Slot"); self.btn_add.clicked.connect(self._add_slot_ui); self.sidebar.addWidget(self.btn_add)
        self.sidebar.addStretch()

        # Matrix
        self.grid_scroll = QtWidgets.QScrollArea(); self.grid_scroll.setWidgetResizable(True); self.grid_container = QtWidgets.QWidget(); self.grid_layout = QtWidgets.QGridLayout(self.grid_container); self.grid_scroll.setWidget(self.grid_container); layout.addWidget(self.grid_scroll)

        self.widgets = {}
        for c in self.state.data["slot_configs"]: self._build_slot_ui(c["id"])
        self._rebuild_grid()

    def _add_url(self):
        t, ok = QtWidgets.QInputDialog.getText(self, "URL", "Enter RTSP/HTTP:"); 
        if ok and t: self.state.add_saved_source(t); self._refresh_srcs()

    def _add_file(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Media",
            "",
            "Media (*.mp4 *.avi *.mkv *.mov *.wmv *.webm *.m4v *.mpeg *.mpg *.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp)"
        )
        if f: self.state.add_saved_source(f); self._refresh_srcs()

    def _add_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.state.add_saved_source(folder)
            self._refresh_srcs()

    def _select_batch_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_path_label.setText(folder)

    def _refresh_srcs(self):
        self.src_list.clear(); self.src_list.addItems(self.state.data["saved_sources"])
        # Update all slot combos
        for i in range(self.slot_area.count()):
            w = self.slot_area.itemAt(i).widget()
            if w:
                cb = w.findChild(QtWidgets.QComboBox)
                if cb: cur = cb.currentText(); cb.clear(); cb.addItems(self.state.data["saved_sources"]); cb.setCurrentText(cur)

    def _add_slot_ui(self): nid = self.state.add_slot(); self._build_slot_ui(nid); self._rebuild_grid()

    def _build_slot_ui(self, sid):
        card = QtWidgets.QFrame(); card.setStyleSheet("background: #333; border-radius: 5px; margin: 2px;"); l = QtWidgets.QVBoxLayout(card)
        l.addWidget(QtWidgets.QLabel(f"<b>Slot #{sid+1}</b>"))
        scb = QtWidgets.QComboBox(); scb.addItems(self.state.data["saved_sources"]); l.addWidget(scb)
        mcb = QtWidgets.QComboBox(); mcb.addItems(get_available_models()); l.addWidget(mcb)
        dcb = QtWidgets.QComboBox()
        for dev_id, label in list_compute_devices():
            dcb.addItem(label, userData=dev_id)
        l.addWidget(dcb)
        device_label = QtWidgets.QLabel("Device: Auto")
        l.addWidget(device_label)
        
        btn = QtWidgets.QPushButton("🚀 Apply")
        btn.clicked.connect(lambda _=False, sid=sid, scb=scb, mcb=mcb, dcb=dcb, label=device_label: self._start_slot(sid, scb, mcb, dcb, label))
        l.addWidget(btn)
        self.slot_area.addWidget(card)
        self.slot_device_labels[sid] = device_label

    def _start_slot(self, sid, scb, mcb, dcb, device_label):
        src = scb.currentText()
        if os.path.isdir(src):
            QtWidgets.QMessageBox.warning(self, "Unsupported", "Папку не можна запускати як live-стрім. Використайте Batch Folder Detection.")
            return
        device_choice = dcb.currentData() or "auto"
        self.slot_device_choices[sid] = device_choice
        self.state.update_slot(sid, {"src": src, "model": mcb.currentText(), "running": True})
        self.streamer.start(src)
        self.engines[sid] = DetectionEngine(
            model_path=None if "Mock" in mcb.currentText() else mcb.currentText(),
            device=device_choice
        )
        device_label.setText(f"Device: {self.engines[sid].device_label}")
        if sid not in self.widgets: self.widgets[sid] = VideoWidget(sid)
        self._rebuild_grid()

    def _rebuild_grid(self):
        for i in reversed(range(self.grid_layout.count())): self.grid_layout.itemAt(i).widget().setParent(None)
        cols = self.state.data["grid_columns"]
        active_ids = [s["id"] for s in self.state.data["slot_configs"] if s.get("running")]
        for idx, sid in enumerate(sorted(active_ids)):
            if sid not in self.widgets:
                self.widgets[sid] = VideoWidget(sid)
            self.grid_layout.addWidget(self.widgets[sid], idx // cols, idx % cols)

    def _grid_change(self, v): self.state.data["grid_columns"] = v; self._rebuild_grid()

    def toggle_web(self):
        if not self.web_process: self.web_process = subprocess.Popen(["streamlit", "run", "app.py"]); self.btn_web.setText("Stop Web")
        else: self.web_process.terminate(); self.web_process = None; self.btn_web.setText("Start Web")

    def _update_loop(self):
        for sid, w in self.widgets.items():
            cfg = next((s for s in self.state.data["slot_configs"] if s["id"] == sid), None)
            if cfg and cfg["running"]:
                f = self.streamer.get_frame(cfg["src"])
                if f is not None:
                    if sid not in self.engines:
                        device_choice = self.slot_device_choices.get(sid, "auto")
                        self.engines[sid] = DetectionEngine(
                            model_path=None if "Mock" in cfg["model"] else cfg["model"],
                            device=device_choice
                        )
                        label = self.slot_device_labels.get(sid)
                        if label:
                            label.setText(f"Device: {self.engines[sid].device_label}")
                    p, _ = self.engines[sid].process_frame(f, night_mode=cfg["night_mode"], quality=cfg["quality"], zoom=cfg["zoom"])
                    w.update_frame(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))

    def _run_batch_folder(self):
        folder = self.folder_path_label.text().strip()
        if not folder or not os.path.isdir(folder):
            QtWidgets.QMessageBox.warning(self, "Folder", "Оберіть папку з файлами.")
            return
        model_path = self.folder_model_cb.currentText()
        self._process_folder_batch(folder, model_path)

    def _process_folder_batch(self, folder, model_path):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm", ".m4v", ".mpeg", ".mpg"}
        files = []
        for root, _, names in os.walk(folder):
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext in image_exts or ext in video_exts:
                    full = os.path.join(root, name)
                    files.append(full)
        files = sorted(files, key=lambda p: os.path.getmtime(p))
        if not files:
            QtWidgets.QMessageBox.information(self, "Folder", "Не знайдено підтримуваних файлів.")
            return

        model_label = "Mock" if "Mock" in model_path else os.path.splitext(os.path.basename(model_path))[0]
        date_stamp = datetime.now().strftime("%Y%m%d")
        default_zip = f"{model_label}_{date_stamp}_detected.zip"
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ZIP", default_zip, "ZIP (*.zip)")
        if not save_path:
            return

        temp_dir = tempfile.mkdtemp(prefix="batch_detect_")
        engine = DetectionEngine(model_path=None if "Mock" in model_path else model_path)
        progress = QtWidgets.QProgressDialog("Processing files...", "Cancel", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        output_files = []
        try:
            for idx, path in enumerate(files, start=1):
                if progress.wasCanceled():
                    break
                ext = os.path.splitext(path)[1].lower()
                rel_dir = os.path.relpath(os.path.dirname(path), folder)
                rel_dir = "" if rel_dir == "." else rel_dir
                target_dir = os.path.join(temp_dir, rel_dir)
                os.makedirs(target_dir, exist_ok=True)
                if ext in image_exts:
                    frame = cv2.imread(path)
                    if frame is None:
                        continue
                    processed, _ = engine.process_frame(frame)
                    out_name = os.path.splitext(os.path.basename(path))[0] + "_detected.jpg"
                    out_path = os.path.join(target_dir, out_name)
                    cv2.imwrite(out_path, processed)
                    output_files.append(out_path)
                else:
                    cap = cv2.VideoCapture(path)
                    if not cap.isOpened():
                        continue
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out_name = os.path.splitext(os.path.basename(path))[0] + "_detected.mp4"
                    out_path = os.path.join(target_dir, out_name)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        processed, _ = engine.process_frame(frame)
                        writer.write(processed)
                    writer.release()
                    cap.release()
                    output_files.append(out_path)
                progress.setValue(idx)
                QtWidgets.QApplication.processEvents()

            with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for full in output_files:
                    arc = os.path.relpath(full, temp_dir)
                    zf.write(full, arc)

            QtWidgets.QMessageBox.information(self, "Done", f"Архів збережено: {save_path}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv); app.setStyle("Fusion")
    window = VisionAdminApp(); window.show(); sys.exit(app.exec())
