import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))

from tracker.byte_tracker import ByteTracker, Track
from utils.reid import AppearanceGallery


def make_frame(color, w=640, h=480):
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def make_det(bbox, conf=0.8):
    return {"bbox": list(map(float, bbox)), "confidence": conf}


def make_person_color_frame(person_color, bbox=(100, 100, 200, 400), bg_color=(128, 128, 128), w=640, h=480):
    frame = np.full((h, w, 3), bg_color, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    frame[y1:y2, x1:x2] = person_color
    return frame


def find_id(tracked, bbox, label):
    for p in tracked:
        if abs(p["bbox"][0] - bbox[0]) < 5 and abs(p["bbox"][2] - bbox[2]) < 5:
            return p["id"]
    raise AssertionError(f"{label} bbox {bbox} not in tracked: {tracked}")


def run_n(tracker, dets, frame, n=3):
    """Feed the same frame n times (default 3) so tracks reach min_hits."""
    out = []
    for _ in range(n):
        out = tracker.update(dets, frame=frame)
    return out


def test_persistence(tmp_path):
    gallery_file = str(tmp_path / "test_gallery.json")
    gallery = AppearanceGallery(data_file=gallery_file)

    tracker1 = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=10, gallery=gallery)
    det = [make_det([100, 100, 200, 400])]
    tracked = run_n(tracker1, det, frame=make_person_color_frame((0, 0, 255)))
    pid1 = found = find_id(tracked, [100, 100, 200, 400], "person1")

    gallery.save_now()

    gallery2 = AppearanceGallery(data_file=gallery_file)
    tracker2 = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=10, gallery=gallery2)
    tracked2 = run_n(tracker2, [make_det([100, 100, 200, 400])], frame=make_person_color_frame((0, 0, 255)))
    pid2 = find_id(tracked2, [100, 100, 200, 400], "person1")

    print(f"[PASS] Persistence: old_id={pid1}, new_id={pid2}")
    assert pid1 == pid2, f"Expected same ID {pid1} but got {pid2}"
    print("[PASS] Person re-identified with same ID after gallery reload!")


def test_reidentification():
    gallery = AppearanceGallery(data_file="data/test_reid.json")
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=3, gallery=gallery)

    f1 = make_person_color_frame((255, 0, 0))
    tracked1 = run_n(tracker, [make_det([100, 100, 200, 400])], frame=f1)
    id1 = find_id(tracked1, [100, 100, 200, 400], "person1")
    print(f"Frame 1: detected person ID={id1}")

    for i in range(6):
        f_blank = make_frame((128, 128, 128))
        tracked_blank = tracker.update([], frame=f_blank)
    assert len(tracked_blank) == 0, f"Expected 0 tracked after person left, got {len(tracked_blank)}"
    print("Frame 2-7: person left frame, 0 tracked")

    f2 = make_person_color_frame((255, 0, 0))
    tracked2 = run_n(tracker, [make_det([100, 100, 200, 400])], frame=f2)
    id2 = find_id(tracked2, [100, 100, 200, 400], "person1")
    print(f"Frame 8: person re-entered, ID={id2}")

    assert id1 == id2, f"Re-ID FAILED: expected {id1} but got {id2}"
    print(f"[PASS] Re-ID SUCCESS: person got same ID {id1} after leaving and re-entering!")

    if os.path.exists("data/test_reid.json"):
        os.remove("data/test_reid.json")


def test_different_persons():
    gallery = AppearanceGallery(data_file="data/test_multi.json", match_threshold=0.60)
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=10, gallery=gallery)

    f_red = make_person_color_frame((255, 0, 0), bbox=(100, 100, 200, 400))
    tracked_red = run_n(tracker, [make_det([100, 100, 200, 400])], frame=f_red)
    id_red = find_id(tracked_red, [100, 100, 200, 400], "red")

    f_blue = make_person_color_frame((0, 0, 255), bbox=(400, 100, 500, 400))
    tracked_blue = run_n(tracker, [make_det([400, 100, 500, 400])], frame=f_blue)
    id_blue = find_id(tracked_blue, [400, 100, 500, 400], "blue")

    assert id_red != id_blue, f"Different persons got same ID: {id_red}"
    print(f"[PASS] Different persons: red={id_red}, blue={id_blue}")

    if os.path.exists("data/test_multi.json"):
        os.remove("data/test_multi.json")


def test_unique_ids():
    gallery = AppearanceGallery(data_file="data/test_unique.json", match_threshold=0.60)
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=10, gallery=gallery)

    persons = []
    bboxes = [
        (100, 100, 200, 400),
        (220, 100, 320, 400),
        (340, 100, 440, 400),
    ]
    frame = np.full((480, 640, 3), (128, 128, 128), dtype=np.uint8)
    for i, bbox in enumerate(bboxes):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        x1, y1, x2, y2 = bbox
        frame[y1:y2, x1:x2] = colors[i]

    tracked = run_n(tracker, [make_det(b) for b in bboxes], frame=frame)
    ids = [find_id(tracked, b, f"p{i}") for i, b in enumerate(bboxes)]
    print(f"[PASS] Unique IDs in one frame: {ids}")
    assert len(set(ids)) == len(ids), f"IDs not unique: {ids}"

    if os.path.exists("data/test_unique.json"):
        os.remove("data/test_unique.json")


def test_two_things_same_color_get_different_ids():
    gallery = AppearanceGallery(data_file="data/test_samecolor.json")
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=10, gallery=gallery)
    frame = np.full((480, 640, 3), (200, 200, 200), dtype=np.uint8)
    frame[100:400, 100:200] = (50, 50, 50)
    frame[100:400, 400:500] = (60, 60, 60)
    det = [make_det([100, 100, 200, 400]), make_det([400, 100, 500, 400])]
    tracked = run_n(tracker, det, frame=frame)
    ids = [find_id(tracked, [100, 100, 200, 400], "dark1"), find_id(tracked, [400, 100, 500, 400], "dark2")]
    print(f"[PASS] Same color frame: ids={ids}")
    assert len(set(ids)) == len(ids), f"Two different people got same ID: {ids}"
    if os.path.exists("data/test_samecolor.json"):
        os.remove("data/test_samecolor.json")


def test_sequential_similar_people_do_not_reuse_id():
    gallery = AppearanceGallery(data_file="data/test_sim.json")
    tracker = ByteTracker(track_thresh=0.4, match_thresh=0.7, track_buffer=3, gallery=gallery)

    f1 = make_person_color_frame((50, 50, 50))
    t1 = run_n(tracker, [make_det([100, 100, 200, 400])], frame=f1)
    id1 = find_id(t1, [100, 100, 200, 400], "p1")
    for _ in range(6):
        tracker.update([], frame=make_frame((200, 200, 200)))

    f2 = make_person_color_frame((75, 75, 75))
    t2 = run_n(tracker, [make_det([100, 100, 200, 400])], frame=f2)
    id2 = find_id(t2, [100, 100, 200, 400], "p2")

    print(f"[PASS] Sequential similar people: p1={id1}, p2={id2}")
    assert id1 != id2, f"Different similar-looking person got reused ID {id1}"
    if os.path.exists("data/test_sim.json"):
        os.remove("data/test_sim.json")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    from pathlib import Path
    import tempfile

    Track._next_id = 1

    print("=" * 60)
    print("TEST 1: Re-identification (leave and re-enter)")
    print("=" * 60)
    test_reidentification()

    print()
    print("=" * 60)
    print("TEST 2: Different persons get different IDs")
    print("=" * 60)
    test_different_persons()

    print()
    print("=" * 60)
    print("TEST 3: Multiple persons in one frame get unique IDs")
    print("=" * 60)
    test_unique_ids()

    print()
    print("=" * 60)
    print("TEST 4: Persistence across gallery reload")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmp:
        test_persistence(Path(tmp))

    Track._next_id = 1
    print()
    print("=" * 60)
    print("TEST 5: Two same-looking people get different IDs")
    print("=" * 60)
    test_two_things_same_color_get_different_ids()

    Track._next_id = 1
    print()
    print("=" * 60)
    print("TEST 6: Sequential similar people do not reuse ID")
    print("=" * 60)
    test_sequential_similar_people_do_not_reuse_id()

    Track._next_id = 1
    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)