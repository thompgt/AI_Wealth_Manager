import json
import os
from typing import Dict, Any

MEMORY_FILE = "user_memory.json"

class UserMemory:
    def __init__(self, file_path: str = MEMORY_FILE):
        self.file_path = file_path
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.data = json.load(f)
            except:
                self.data = {}
        else:
            self.data = {}

    def _save_memory(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.data.get(user_id, {})

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
        if user_id not in self.data:
            self.data[user_id] = {
                "created_at": "2024-01-01", 
                "history": [],
                "preferences": {}
            }
        
        # Deep merge or simple update? Simple for now.
        for k, v in updates.items():
            self.data[user_id][k] = v
        
        self._save_memory()

    def add_interaction(self, user_id: str, prompt: str, response: str):
        if user_id not in self.data:
            self.update_user_profile(user_id, {})
        
        if "history" not in self.data[user_id]:
            self.data[user_id]["history"] = []
            
        self.data[user_id]["history"].append({
            "prompt": prompt,
            "response": response
        })
        self._save_memory()

# Singleton
memory = UserMemory()
