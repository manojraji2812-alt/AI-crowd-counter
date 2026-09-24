import json
import os
import time
import threading
import numpy as np
import cv2


class PersonAppearance:
    def __init__(self, person_id, max_gallery=12):
        self.person_id = person_id
        self.max_gallery = max_gallery
        self.gallery = []
        self.avg_feature = None
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.detection_count = 0

    def add(self, feature):
        if feature is None:
            return
        self.gallery.append(feature.copy())
        if len(self.gallery) > self.max_gallery:
            self.gallery = self.gallery[-self.max_gallery:]
        self.last_seen = time.time()
        self.detection_count += 1
        self._update_avg()

    def _update_avg(self):
        if not self.gallery:
            self.avg_feature = None
            return
        n = len(self.gallery)
        weights = np.linspace(0.3, 1.0, n)
        weights /= weights.sum()
        stacked = np.stack(self.gallery, axis=0)
        self.avg_feature = np.average(stacked, axis=0, weights=weights)

    def similarity(self, feature):
        if self.avg_feature is None or feature is None:
            return 0.0
        a = self.avg_feature.flatten().astype(np.float64)
        b = feature.flatten().astype(np.float64)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def to_dict(self):
        return {
            "id": self.person_id,
            "gallery": [f.tolist() for f in self.gallery[-self.max_gallery:]],
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "detection_count": self.detection_count,
        }

    @classmethod
    def from_dict(cls, data):
        p = cls(data["id"])
        p.gallery = [np.array(f, dtype=np.float64) for f in data.get("gallery", [])]
        p.first_seen = data.get("first_seen", time.time())
        p.last_seen = data.get("last_seen", time.time())
        p.detection_count = data.get("detection_count", 0)
        p._update_avg()
        return p


class AppearanceGallery:
    SAVE_INTERVAL = 30
    RETENTION_SECONDS = 86400

    def __init__(self, data_file="data/reid_gallery.json", match_threshold=0.72):
        self.data_file = data_file
        self.match_threshold = match_threshold
        self.persons = {}
        self.lock = threading.Lock()
        self.last_save = time.time()
        self._load()

    def match_or_create(self, feature, min_sim=None, margin=0.06):
        min_sim = self.match_threshold if min_sim is None else min_sim
        with self.lock:
            if feature is None:
                return None, False

            best_person = None
            best_score = -1.0
            second_score = -1.0

            for person in self.persons.values():
                score = person.similarity(feature)
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_person = person
                elif score > second_score:
                    second_score = score

            if (best_person is not None and best_score >= min_sim
                    and best_score - second_score >= margin):
                best_person.add(feature)
                self._auto_save()
                return best_person.person_id, True

            new_id = self._next_id()
            new_person = PersonAppearance(new_id)
            new_person.add(feature)
            self.persons[new_id] = new_person
            self._auto_save()
            return new_id, False

    def match(self, feature, min_sim=None, margin=0.06):
        min_sim = self.match_threshold if min_sim is None else min_sim
        with self.lock:
            if feature is None:
                return None
            best_person = None
            best_score = -1.0
            second_score = -1.0
            for person in self.persons.values():
                score = person.similarity(feature)
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_person = person
                elif score > second_score:
                    second_score = score
            if best_person is None or best_score < min_sim:
                return None
            if best_score - second_score < margin:
                return None
            return best_person.person_id

    def register(self, person_id, feature):
        with self.lock:
            if feature is None:
                return
            if person_id in self.persons:
                self.persons[person_id].add(feature)
            else:
                p = PersonAppearance(person_id)
                p.add(feature)
                self.persons[person_id] = p
            self._auto_save()

    def get_known_count(self):
        with self.lock:
            return len(self.persons)

    def get_max_id(self):
        with self.lock:
            if not self.persons:
                return 0
            return max(self.persons.keys())

    def _next_id(self):
        if not self.persons:
            return 1
        return max(self.persons.keys()) + 1

    def _auto_save(self):
        now = time.time()
        if now - self.last_save < self.SAVE_INTERVAL:
            return
        self.last_save = now
        self._save()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            data = {}
            now = time.time()
            for pid, p in self.persons.items():
                if now - p.last_seen <= self.RETENTION_SECONDS:
                    data[str(pid)] = p.to_dict()
            with open(self.data_file, "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, "r") as f:
                    raw = json.load(f)
                now = time.time()
                for pid_str, pdata in raw.items():
                    p = PersonAppearance.from_dict(pdata)
                    if now - p.last_seen <= self.RETENTION_SECONDS:
                        self.persons[p.person_id] = p
                print(f"[INFO] Loaded {len(self.persons)} known persons from gallery.")
        except Exception:
            self.persons = {}

    def save_now(self):
        with self.lock:
            self._save()
