import time
import threading
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class TrackState:
    NEW = 0
    TRACKED = 1
    LOST = 2


class Track:
    _next_id = 1

    @classmethod
    def seed_next_id(cls, max_id):
        if max_id is not None and max_id >= cls._next_id:
            cls._next_id = max_id + 1

    def __init__(self, bbox, score, appearance=None, track_id=None):
        if track_id is not None:
            self.id = track_id
        else:
            self.id = Track._next_id
            Track._next_id += 1
        self.bbox = bbox
        self.score = score
        self.state = TrackState.NEW
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.lost_time = None
        self.appearance_gallery = []
        self.avg_appearance = None
        self.velocity = np.zeros(2)
        self._prev_center = None
        if appearance is not None:
            self.appearance_gallery.append(appearance.copy())
            self.avg_appearance = appearance.copy()

    @staticmethod
    def _center(bbox):
        return np.array([
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        ], dtype=np.float64)

    def predict(self):
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox, score, appearance=None):
        new_box = np.asarray(bbox, dtype=np.float64)
        old_box = np.asarray(self.bbox, dtype=np.float64)
        center_move = float(np.hypot(
            (new_box[0] + new_box[2]) / 2 - (old_box[0] + old_box[2]) / 2,
            (new_box[1] + new_box[3]) / 2 - (old_box[1] + old_box[3]) / 2,
        ))
        if center_move < 12.0:
            alpha = 0.6
        elif center_move < 40.0:
            alpha = 0.85
        else:
            alpha = 0.95
        smoothed = new_box * alpha + old_box * (1 - alpha)
        self.bbox = [float(v) for v in smoothed]

        new_center = self._center(self.bbox)
        if self._prev_center is not None:
            self.velocity = 0.7 * self.velocity + 0.3 * (new_center - self._prev_center)
        self._prev_center = new_center

        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED
        self.lost_time = None
        if appearance is not None:
            self.appearance_gallery.append(appearance.copy())
            if len(self.appearance_gallery) > 30:
                self.appearance_gallery = self.appearance_gallery[-30:]
            self._recompute_avg()

    def mark_lost(self):
        self.state = TrackState.LOST
        self.lost_time = time.time()

    def _recompute_avg(self):
        if not self.appearance_gallery:
            self.avg_appearance = None
            return
        weights = np.linspace(0.3, 1.0, len(self.appearance_gallery))
        weights /= weights.sum()
        stacked = np.stack(self.appearance_gallery, axis=0)
        self.avg_appearance = np.average(stacked, axis=0, weights=weights)

    def get_gallery_avg(self, n=5):
        if not self.appearance_gallery:
            return self.avg_appearance
        recent = self.appearance_gallery[-n:]
        return np.mean(recent, axis=0)


class ByteTracker:
    def __init__(self, track_thresh=0.4, match_thresh=0.7, track_buffer=150, gallery=None,
                 reidentify_appearance_thresh=0.82, reidentify_margin=0.08,
                 lost_recall_seconds=120.0, output_keep_frames=None, min_hits=3):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracks = []
        self.lost_tracks = []
        self.max_lost = 1000
        self.reidentify_appearance_thresh = reidentify_appearance_thresh
        self.reidentify_margin = reidentify_margin
        self.lost_recall_seconds = lost_recall_seconds
        self.output_keep_frames = output_keep_frames if output_keep_frames is not None else min(track_buffer, 10)
        self.min_hits = min_hits
        self.gallery = gallery
        self.lock = threading.RLock()

    def update(self, detections, frame=None):
        with self.lock:
            return self._update_locked(detections, frame)

    def _update_locked(self, detections, frame=None):
        appearances = []
        for det in detections:
            app = self._extract_appearance(det["bbox"], frame) if frame is not None else None
            appearances.append(app)

        for track in self.tracks:
            track.predict()

        matched, unmatched_dets, unmatched_trks = self._match(detections, self.tracks, appearances)

        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(
                detections[det_idx]["bbox"],
                detections[det_idx]["confidence"],
                appearances[det_idx],
            )
            if self.gallery is not None and appearances[det_idx] is not None:
                self.gallery.register(self.tracks[trk_idx].id, appearances[det_idx])

        active_ids = {t.id for t in self.tracks}
        claimed_ids = set(active_ids)

        new_tracks = []
        for det_idx in unmatched_dets:
            if detections[det_idx]["confidence"] < self.track_thresh:
                continue

            reidentified = self._try_reidentify(
                detections[det_idx]["bbox"],
                appearances[det_idx],
                frame,
                claimed_ids,
            )
            if reidentified is not None:
                if reidentified.id in claimed_ids:
                    reidentified = None
                else:
                    claimed_ids.add(reidentified.id)
                    reidentified.update(
                        detections[det_idx]["bbox"],
                        detections[det_idx]["confidence"],
                        appearances[det_idx],
                    )
                    new_tracks.append(reidentified)
                    if self.gallery is not None and appearances[det_idx] is not None:
                        self.gallery.register(reidentified.id, appearances[det_idx])
            if reidentified is None:
                track = Track(
                    detections[det_idx]["bbox"],
                    detections[det_idx]["confidence"],
                    appearances[det_idx],
                )
                claimed_ids.add(track.id)
                new_tracks.append(track)
                if self.gallery is not None and appearances[det_idx] is not None:
                    self.gallery.register(track.id, appearances[det_idx])
        self.tracks.extend(new_tracks)

        alive = []
        for track in self.tracks:
            if track.time_since_update > self.track_buffer:
                track.mark_lost()
                self.lost_tracks.append(track)
            else:
                alive.append(track)
        self.tracks = alive

        self.lost_tracks = self.lost_tracks[-self.max_lost:]

        tracked_persons = []
        seen_ids = set()
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update <= self.output_keep_frames:
                if track.id in seen_ids:
                    new_id = Track._next_id
                    Track._next_id += 1
                    track.id = new_id
                seen_ids.add(track.id)
                tracked_persons.append({
                    "id": track.id,
                    "bbox": track.bbox,
                    "confidence": track.score,
                })
        return tracked_persons

    def _try_reidentify(self, bbox, appearance, frame=None, claimed_ids=None):
        claimed_ids = claimed_ids or set()
        if appearance is None:
            return None

        self._cull_lost_tracks()

        best_lost = None
        best_lost_sim = -1.0
        second_lost_sim = -1.0
        now = time.time()

        for track in self.lost_tracks:
            if track.id in claimed_ids:
                continue
            if track.lost_time is not None and (now - track.lost_time) > self.lost_recall_seconds:
                continue

            ref_appearance = track.get_gallery_avg(5)
            if ref_appearance is None:
                continue

            appearance_sim = self._cosine_similarity(appearance, ref_appearance)
            if appearance_sim > best_lost_sim:
                second_lost_sim = best_lost_sim
                best_lost_sim = appearance_sim
                best_lost = track
            elif appearance_sim > second_lost_sim:
                second_lost_sim = appearance_sim

        if (best_lost is not None
                and best_lost_sim >= self.reidentify_appearance_thresh
                and (best_lost_sim - second_lost_sim) >= self.reidentify_margin):
            self.lost_tracks.remove(best_lost)
            best_lost.state = TrackState.TRACKED
            return best_lost

        if self.gallery is not None:
            known_id = self.gallery.match(
                appearance,
                min_sim=self.reidentify_appearance_thresh,
                margin=self.reidentify_margin,
            )
            if known_id is not None and known_id not in claimed_ids:
                for track in self.lost_tracks:
                    if track.id == known_id:
                        self.lost_tracks.remove(track)
                        track.state = TrackState.TRACKED
                        return track
                new_track = Track(bbox, 1.0, appearance, track_id=known_id)
                if known_id >= Track._next_id:
                    Track._next_id = known_id + 1
                return new_track

        return None

    def _cull_lost_tracks(self):
        now = time.time()
        kept = []
        for track in self.lost_tracks:
            if track.lost_time is not None and (now - track.lost_time) > self.lost_recall_seconds:
                continue
            kept.append(track)
        self.lost_tracks = kept

    def _match(self, detections, tracks, appearances=None):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))

        cost_matrix = np.zeros((len(detections), len(tracks)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(tracks):
                iou_dist = self._iou_distance(det["bbox"], trk.bbox)
                center = self._center_distance(det["bbox"], trk.bbox)
                center_dist = min(center / 500.0, 1.0)

                app_cost = 0
                app_sim = 0.0
                if (appearances is not None and d < len(appearances)
                        and appearances[d] is not None and trk.avg_appearance is not None):
                    app_sim = self._cosine_similarity(appearances[d], trk.avg_appearance)
                    app_cost = 1.0 - app_sim

                iou = ByteTracker._compute_iou(det["bbox"], trk.bbox)
                spatial_ok = iou > 0.1 or (center < 60.0 and app_sim > 0.6)
                if not spatial_ok:
                    cost_matrix[d, t] = 1.2
                    continue

                cost_matrix[d, t] = iou_dist * 0.4 + center_dist * 0.2 + app_cost * 0.4

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < self.match_thresh:
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _cosine_similarity(a, b):
        if a is None or b is None:
            return 0.0
        try:
            a_flat = a.flatten().astype(np.float64)
            b_flat = b.flatten().astype(np.float64)
            norm_a = np.linalg.norm(a_flat)
            norm_b = np.linalg.norm(b_flat)
            if norm_a < 1e-6 or norm_b < 1e-6:
                return 0.0
            return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _extract_appearance(self, bbox, frame):
        if frame is None:
            return None
        try:
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            resized = cv2.resize(crop, (32, 64))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

            hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            cv2.normalize(hist_v, hist_v)

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            hist_gray = cv2.calcHist([gray], [0], None, [16], [0, 256])
            cv2.normalize(hist_gray, hist_gray)

            ch, cw = resized.shape[:2]
            upper = resized[:ch // 3, :]
            middle = resized[ch // 3:2 * ch // 3, :]
            lower = resized[2 * ch // 3:, :]

            hist_upper = self._region_histogram(upper)
            hist_middle = self._region_histogram(middle)
            hist_lower = self._region_histogram(lower)

            features = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten(),
                hist_gray.flatten(),
                hist_upper.flatten(),
                hist_middle.flatten(),
                hist_lower.flatten(),
            ]).astype(np.float64)

            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features
        except Exception:
            return None

    def _region_histogram(self, region):
        if region.size == 0:
            return np.zeros(36)
        try:
            small = cv2.resize(region, (16, 16))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [20], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            cv2.normalize(hist_h, hist_h)
            cv2.normalize(hist_s, hist_s)
            return np.concatenate([hist_h.flatten(), hist_s.flatten()])
        except Exception:
            return np.zeros(36)

    @staticmethod
    def _center_distance(bbox_a, bbox_b):
        cx_a = (bbox_a[0] + bbox_a[2]) / 2
        cy_a = (bbox_a[1] + bbox_a[3]) / 2
        cx_b = (bbox_b[0] + bbox_b[2]) / 2
        cy_b = (bbox_b[1] + bbox_b[3]) / 2
        return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

    @staticmethod
    def _iou_distance(bbox_a, bbox_b):
        iou = ByteTracker._compute_iou(bbox_a, bbox_b)
        return 1 - iou

    @staticmethod
    def _compute_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0
