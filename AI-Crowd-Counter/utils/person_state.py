import json
import os
import time


class PersonStateManager:
    THRESHOLD_SECONDS = 60
    RETENTION_SECONDS = 86400  # 24 hours
    SAVE_INTERVAL = 30

    def __init__(self, data_file="data/persons.json"):
        self.persons = {}
        self.data_file = data_file
        self.last_save = time.time()
        self._load()

    def update(self, tracked_persons: list):
        active_ids = set()
        for person in tracked_persons:
            pid = person["id"]
            active_ids.add(pid)
            if pid not in self.persons:
                self.persons[pid] = {
                    "id": pid,
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "currently_visible": True,
                    "continuous_start": time.time(),
                }
            else:
                self.persons[pid]["last_seen"] = time.time()
                if not self.persons[pid]["currently_visible"]:
                    self.persons[pid]["continuous_start"] = time.time()
                self.persons[pid]["currently_visible"] = True

        for pid in self.persons:
            if pid not in active_ids:
                self.persons[pid]["currently_visible"] = False

        self._auto_save()
        self._cleanup()

    def _person_entry(self, pid, info, now):
        continuous_duration = now - info["continuous_start"] if info["currently_visible"] else 0
        total_duration = now - info["first_seen"]
        return {
            "id": pid,
            "first_seen": info["first_seen"],
            "last_seen": info["last_seen"],
            "duration": total_duration,
            "continuous_duration": continuous_duration,
            "currently_visible": info["currently_visible"],
        }

    def get_active_persons(self):
        """Persons with >=60s total duration, shown for 24 hours."""
        now = time.time()
        result = []
        for pid, info in self.persons.items():
            if now - info["last_seen"] > self.RETENTION_SECONDS:
                continue
            total_duration = now - info["first_seen"]
            if total_duration >= self.THRESHOLD_SECONDS:
                result.append(self._person_entry(pid, info, now))
        return sorted(result, key=lambda x: x["id"])

    def get_live_persons(self):
        """Persons currently visible/tracked in the live feed right now."""
        now = time.time()
        result = []
        for pid, info in self.persons.items():
            if info["currently_visible"]:
                result.append(self._person_entry(pid, info, now))
        return sorted(result, key=lambda x: x["id"])

    def get_all_persons(self):
        """Every person seen within the last 24 hours."""
        now = time.time()
        result = []
        for pid, info in self.persons.items():
            if now - info["last_seen"] > self.RETENTION_SECONDS:
                continue
            result.append(self._person_entry(pid, info, now))
        return sorted(result, key=lambda x: x["id"])

    def get_current_count(self):
        return sum(1 for p in self.persons.values() if p["currently_visible"])

    def _cleanup(self):
        now = time.time()
        expired = [pid for pid, info in self.persons.items()
                   if now - info["last_seen"] > self.RETENTION_SECONDS]
        for pid in expired:
            del self.persons[pid]

    def _auto_save(self):
        now = time.time()
        if now - self.last_save < self.SAVE_INTERVAL:
            return
        self.last_save = now
        self._save()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            with open(self.data_file, "w") as f:
                json.dump(self.persons, f)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    raw = json.load(f)
                self.persons = {}
                for k, v in raw.items():
                    self.persons[int(k)] = v
                self._cleanup()
                print(f"[INFO] Loaded {len(self.persons)} persons from disk.")
        except Exception:
            self.persons = {}

    def save_now(self):
        self._save()
