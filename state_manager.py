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
            "web_users_auth": {"admin": "admin123"}, # login: password
            "user_access": {}, # login: status (pending/allowed/blocked)
            "slot_configs": [
                {"id": 0, "src": "0", "model": "Mock (No Model)", "mode": "Live", "conf": 0.5, "quality": "480p", "visibility_map": {}, "night_mode": False, "running": False}
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
                    
                    # Migration: Перевірка наявності 'id' у завантажених конфігураціях
                    if "slot_configs" in saved:
                        for idx, slot in enumerate(saved["slot_configs"]):
                            if "id" not in slot:
                                slot["id"] = idx
                        
                        # Оновлення next_slot_id на основі завантажених даних
                        if saved["slot_configs"]:
                            saved["next_slot_id"] = max(s["id"] for s in saved["slot_configs"]) + 1
                    
                    self.data.update(saved)
            except Exception as e:
                print(f"Error loading state: {e}")

    def save(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def add_slot(self):
        new_id = self.data["next_slot_id"]
        self.data["slot_configs"].append({
            "id": new_id, "src": "0", "model": "Mock (No Model)", "mode": "Live", 
            "conf": 0.5, "quality": "480p", "visibility_map": {}, "night_mode": False, "running": False
        })
        self.data["next_slot_id"] += 1
        self.save()
        return new_id

    def remove_slot(self, slot_id):
        self.data["slot_configs"] = [s for s in self.data["slot_configs"] if s["id"] != slot_id]
        self.save()

    def update_slot(self, slot_id, config):
        for s in self.data["slot_configs"]:
            if s["id"] == slot_id:
                s.update(config)
                break
        self.save()

    def manage_user(self, login, action):
        if action == "delete":
            if login in self.data["user_access"]: del self.data["user_access"][login]
            if login in self.data["web_users_auth"]: del self.data["web_users_auth"][login]
        else:
            self.data["user_access"][login] = action # allowed / blocked
        self.save()
