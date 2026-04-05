import sys
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_PLUGIN_PATH"] = "" 

import cv2
import threading
import signal
import subprocess
import time
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from engine import DetectionEngine, get_available_models
from streamer import MultiCameraStreamer, scan_cameras
from state_manager import StateManager

class VideoWidget(QtWidgets.QLabel):
    def __init__(self, slot_id, parent=None):
        super().__init__(parent)
        self.slot_id = slot_id
        self.setMinimumSize(480, 270)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333; border-radius: 8px;")
        self.setText(f"Камера #{slot_id + 1}")
        self.setScaledContents(False)

    def update_frame(self, frame):
        if frame is None: return
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(q_img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

class UserTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Користувач", "Статус", "Дії"])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.state = StateManager()

    def refresh(self):
        self.setRowCount(0)
        # Auth: User Access Status
        for login, status in self.state.data["user_access"].items():
            row = self.rowCount()
            self.insertRow(row)
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(login))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(status.upper()))
            
            btn_panel = QtWidgets.QWidget()
            l = QtWidgets.QHBoxLayout(btn_panel)
            l.setContentsMargins(0, 0, 0, 0)
            
            btn_allow = QtWidgets.QPushButton("Allow")
            btn_allow.clicked.connect(lambda _, log=login: self.state.manage_user(log, "allowed"))
            btn_block = QtWidgets.QPushButton("Block")
            btn_block.clicked.connect(lambda _, log=login: self.state.manage_user(log, "blocked"))
            btn_del = QtWidgets.QPushButton("X")
            btn_del.clicked.connect(lambda _, log=login: self.state.manage_user(log, "delete"))
            
            l.addWidget(btn_allow); l.addWidget(btn_block); l.addWidget(btn_del)
            self.setCellWidget(row, 2, btn_panel)

class VisionAdminApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Intelligence v4 Pro")
        self.resize(1600, 1000)
        self.state = StateManager()
        self.streamer = MultiCameraStreamer()
        self.engines = {}
        self.web_process = None
        self._init_ui()
        self.timer = QTimer(); self.timer.timeout.connect(self._update_loop); self.timer.start(30)
        self.list_timer = QTimer(); self.list_timer.timeout.connect(self._update_ui_lists); self.list_timer.start(2000)

    def _init_ui(self):
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # 🛠️ LEFT: Controls Sidebar
        scroll = QtWidgets.QScrollArea(); scroll.setFixedWidth(380); scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)
        ctrl_panel = QtWidgets.QWidget(); scroll.setWidget(ctrl_panel)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)

        # Group: WEB Server
        web_group = QtWidgets.QGroupBox("🌐 Web Control Node")
        web_l = QtWidgets.QVBoxLayout(web_group)
        self.btn_web = QtWidgets.QPushButton("🚀 Start Web Server")
        self.btn_web.clicked.connect(self.toggle_web_server)
        web_l.addWidget(self.btn_web)
        self.lbl_web = QtWidgets.QLabel("Status: Offline")
        web_l.addWidget(self.lbl_web); ctrl_layout.addWidget(web_group)

        # Group: Users (Table)
        user_group = QtWidgets.QGroupBox("👤 Web Users Control")
        user_l = QtWidgets.QVBoxLayout(user_group)
        self.user_table = UserTable()
        user_l.addWidget(self.user_table)
        ctrl_layout.addWidget(user_group)

        # Group: Camera Sources
        src_group = QtWidgets.QGroupBox("💾 Video Sources")
        src_l = QtWidgets.QVBoxLayout(src_group)
        self.src_list = QtWidgets.QListWidget()
        self.src_list.addItems(self.state.data["saved_sources"])
        src_l.addWidget(self.src_list)
        btn_add_src = QtWidgets.QPushButton("+ New Source")
        btn_add_src.clicked.connect(self._add_source)
        src_l.addWidget(btn_add_src)
        ctrl_layout.addWidget(src_group)

        # Group: Config Slots
        self.slot_scroll = QtWidgets.QScrollArea(); self.slot_scroll.setFixedHeight(450)
        self.slot_scroll.setWidgetResizable(True)
        self.slot_container = QtWidgets.QWidget()
        self.slot_layout = QtWidgets.QVBoxLayout(self.slot_container)
        self.slot_scroll.setWidget(self.slot_container)
        ctrl_layout.addWidget(self.slot_scroll)

        self.btn_add_slot = QtWidgets.QPushButton("➕ Додати Новий Потік (Slot)")
        self.btn_add_slot.clicked.connect(self.add_new_slot_ui)
        ctrl_layout.addWidget(self.btn_add_slot)

        # 📺 RIGHT: Main Scrollable Grid
        self.grid_scroll = QtWidgets.QScrollArea()
        self.grid_scroll.setWidgetResizable(True)
        self.grid_container = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_container)
        self.grid_scroll.setWidget(self.grid_container)
        main_layout.addWidget(self.grid_scroll)

        self.video_widgets = {}
        for config in self.state.data["slot_configs"]:
            self._build_slot_config_ui(config["id"])

    def add_new_slot_ui(self):
        new_id = self.state.add_slot()
        self._build_slot_config_ui(new_id)

    def _build_slot_config_ui(self, slot_id):
        # UI Slot Card in Sidebar
        card = QtWidgets.QFrame(); card.setFrameShape(QtWidgets.QFrame.StyledPanel)
        card.setStyleSheet("background-color: #333; margin-bottom: 5px; border-radius: 5px;")
        l = QtWidgets.QVBoxLayout(card)
        l.addWidget(QtWidgets.QLabel(f"<b>Вікно #{slot_id + 1}</b>"))
        
        src_cb = QtWidgets.QComboBox(); src_cb.addItems(self.state.data["saved_sources"])
        mod_cb = QtWidgets.QComboBox(); mod_cb.addItems(get_available_models())
        qual_cb = QtWidgets.QComboBox(); qual_cb.addItems(["Native", "720p", "480p", "360p"])
        
        l.addWidget(QtWidgets.QLabel("Source / Model:")); l.addWidget(src_cb); l.addWidget(mod_cb)
        
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Quality:")); h.addWidget(qual_cb)
        l.addLayout(h)

        night_cb = QtWidgets.QCheckBox("🌙 Night Boost (CLAHE)")
        l.addWidget(night_cb)

        # Visibility List
        l.addWidget(QtWidgets.QLabel("Видимість Об'єктів:"))
        vis_list = QtWidgets.QListWidget()
        vis_list.setFixedHeight(100)
        vis_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        l.addWidget(vis_list)

        btn_start = QtWidgets.QPushButton("🚀 Start Stream")
        btn_start.clicked.connect(lambda: self.start_slot(slot_id, src_cb, mod_cb, qual_cb, night_cb, vis_list))
        l.addWidget(btn_start)

        self.slot_layout.insertWidget(self.slot_layout.count(), card)
        
        # Build Widget in Grid
        vw = VideoWidget(slot_id)
        self.video_widgets[slot_id] = vw
        row, col = len(self.video_widgets)//2, len(self.video_widgets)%2
        self.grid_layout.addWidget(vw, row, col)

    def start_slot(self, slot_id, src_cb, mod_cb, qual_cb, night_cb, vis_list):
        src, mod, qual = src_cb.currentText(), mod_cb.currentText(), qual_cb.currentText()
        night = night_cb.isChecked()
        
        if slot_id not in self.engines:
            m_path = None if "Mock" in mod else mod
            self.engines[slot_id] = DetectionEngine(model_path=m_path)
            # Fill visibility map first time
            for cls in self.engines[slot_id].get_all_classes():
                item = QtWidgets.QListWidgetItem(cls)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                vis_list.addItem(item)
        
        vis_map = {}
        for i in range(vis_list.count()):
            it = vis_list.item(i)
            vis_map[it.text()] = (it.checkState() == Qt.Checked)

        self.state.update_slot(slot_id, {
            "src": src, "model": mod, "quality": qual, "night_mode": night,
            "visibility_map": vis_map, "running": True
        })
        self.streamer.start(src)

    def toggle_web_server(self):
        if not self.web_process:
            self.web_process = subprocess.Popen(["streamlit", "run", "app.py"])
            self.state.set_web_status(True)
        else:
            self.web_process.terminate(); self.web_process = None
            self.state.set_web_status(False)

    def _add_source(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "New Source", "Enter IP/DNS/URL:")
        if ok and text:
            self.state.data["saved_sources"].append(text); self.state.save()
            self._update_ui_lists()

    def _update_ui_lists(self):
        self.src_list.clear(); self.src_list.addItems(self.state.data["saved_sources"])
        self.user_table.refresh()

    def _update_loop(self):
        for slot_id, vw in self.video_widgets.items():
            config = next((s for s in self.state.data["slot_configs"] if s["id"] == slot_id), None)
            if config and config["running"]:
                frame = self.streamer.get_frame(config["src"])
                if frame is not None:
                    processed, _ = self.engines[slot_id].process_frame(
                        frame, night_mode=config["night_mode"], visibility_map=config["visibility_map"]
                    )
                    vw.update_frame(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    def closeEvent(self, event):
        self.streamer.stop_all()
        if self.web_process: self.web_process.terminate()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv); app.setStyle("Fusion")
    window = VisionAdminApp(); window.show(); sys.exit(app.exec())
