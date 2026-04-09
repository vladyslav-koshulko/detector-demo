import json
import os
import threading
import time

class StateManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.state_file = "system_state.json"
        self.data = {
            "web_server_running": False,
            "web_users_auth": {"admin": "admin123", "user": "user123"},
            "user_sessions": {}, 
            "grid_columns": 2,
            "slot_configs": [
                {"id": 0, "src": "0", "model": "Mock (No Model)", "mode": "Live", "conf": 0.5, "quality": "480p", "visibility_map": {}, "night_mode": False, "running": False, "zoom": 1.0}
            ],
            "saved_sources": ["0", "1"],
            "next_slot_id": 1
        }
        self.load()

    def load(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    saved = json.load(f)
                    # Migration
                    if "slot_configs" in saved:
                        for idx, slot in enumerate(saved["slot_configs"]):
                            if "id" not in slot: slot["id"] = idx
                    self.data.update(saved)
            except: pass
        if not self.data.get("web_users_auth"):
            self.data["web_users_auth"] = {"admin": "admin123", "user": "user123"}
            self.save()

    def save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def add_saved_source(self, src):
        if src not in self.data["saved_sources"]:
            self.data["saved_sources"].append(src)
            self.save()

    def add_slot(self):
        new_id = self.data["next_slot_id"]
        self.data["slot_configs"].append({
            "id": new_id, "src": "0", "model": "Mock (No Model)", "mode": "Live", 
            "conf": 0.5, "quality": "480p", "visibility_map": {}, "night_mode": False, "running": False, "zoom": 1.0
        })
        self.data["next_slot_id"] += 1
        self.save()
        return new_id

    def update_slot(self, slot_id, config):
        for s in self.data["slot_configs"]:
            if s["id"] == slot_id:
                s.update(config)
                break
        self.save()

    def manage_user(self, login, action, role=None):
        if login not in self.data["user_sessions"]:
            self.data["user_sessions"][login] = {"status": "active", "role": "view"}
        if action == "login":
            self.data["user_sessions"][login]["status"] = "active"
        elif action == "block": self.data["user_sessions"][login]["status"] = "blocked"
        elif action == "set_role": self.data["user_sessions"][login]["role"] = role
        elif action == "delete": 
            if login in self.data["user_sessions"]: del self.data["user_sessions"][login]
        self.save()
