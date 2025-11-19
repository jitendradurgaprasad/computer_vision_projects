import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
import math
from datetime import datetime


# ===========================
# CONFIG
# ===========================
MODEL_PATH = "yolov8n.pt"
EXCEL_PATH = "vehicle_log.xlsx"

DISTANCE_THRESHOLD = 70
MAX_MISSES = 7
MIN_TRACKED_FRAMES = 3
MIN_BBOX_AREA = 350
LINE_OFFSET = 0


# ---------------------------
# Night enhancer
# ---------------------------
def enhance_night(frame, gamma=1.3):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    frame = cv2.LUT(frame, table)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab = cv2.merge([L2, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------
# Excel initialization
# ---------------------------
def init_excel():
    if not os.path.exists(EXCEL_PATH):
        df = pd.DataFrame(columns=["timestamp", "vehicle_id", "direction"])
        df.to_excel(EXCEL_PATH, index=False)


def log_buffer_to_excel(buffer):
    df = pd.read_excel(EXCEL_PATH)
    for tid, direction in buffer:
        df.loc[len(df)] = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tid,
            direction,
        ]
    df.to_excel(EXCEL_PATH, index=False)


# ---------------------------
# Centroid Tracker
# ---------------------------
class CentroidTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        self.up_total = 0
        self.down_total = 0

    def _centroid(self, bb):
        x1, y1, x2, y2 = bb
        return (x1 + x2) // 2, (y1 + y2) // 2

    def update(self, detections):
        centers = []
        bboxes = []

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            if (x2 - x1) * (y2 - y1) < MIN_BBOX_AREA:
                continue

            centers.append(self._centroid(d["bbox"]))
            bboxes.append(d["bbox"])

        assigned_dets = set()
        assigned_tracks = set()

        if centers and self.tracks:
            track_ids = list(self.tracks.keys())
            track_centers = [self.tracks[t]["centroid"] for t in track_ids]

            dist_mat = np.zeros((len(track_centers), len(centers)), dtype=np.float32)
            for i, tc in enumerate(track_centers):
                for j, dc in enumerate(centers):
                    dist_mat[i][j] = math.dist(tc, dc)

            for _ in range(dist_mat.size):
                i, j = np.unravel_index(dist_mat.argmin(), dist_mat.shape)
                if dist_mat[i][j] > DISTANCE_THRESHOLD:
                    break

                tid = track_ids[i]
                self.tracks[tid]["bbox"] = bboxes[j]
                self.tracks[tid]["centroid"] = centers[j]
                self.tracks[tid]["misses"] = 0
                self.tracks[tid]["history"].append(centers[j])
                self.tracks[tid]["frames_seen"] += 1

                assigned_dets.add(j)
                assigned_tracks.add(tid)

                dist_mat[i, :] = 1e6
                dist_mat[:, j] = 1e6

        for j, c in enumerate(centers):
            if j not in assigned_dets:
                tid = self.next_id
                self.next_id += 1

                self.tracks[tid] = {
                    "bbox": bboxes[j],
                    "centroid": centers[j],
                    "misses": 0,
                    "history": [centers[j]],
                    "counted": False,
                    "frames_seen": 1,
                }

        for tid in list(self.tracks.keys()):
            if tid not in assigned_tracks:
                self.tracks[tid]["misses"] += 1
                if self.tracks[tid]["misses"] > MAX_MISSES:
                    del self.tracks[tid]

        output = []
        for tid, info in self.tracks.items():
            output.append({
                "id": tid,
                "bbox": info["bbox"],
                "centroid": info["centroid"],
                "history": info["history"],
                "counted": info["counted"],
            })

        return output

    def mark_counted(self, tid, direction):
        if tid in self.tracks and not self.tracks[tid]["counted"]:
            self.tracks[tid]["counted"] = True
            if direction == "UP":
                self.up_total += 1
            else:
                self.down_total += 1
            return True
        return False


# ---------------------------
# YOLO DETECTOR
# ---------------------------
class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.keep = ["car", "truck", "bus", "motorcycle"]

    def detect(self, frame):
        res = self.model(frame)[0]
        dets = []
        for box in res.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            if label not in self.keep:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({"bbox": (x1, y1, x2, y2)})
        return dets


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🚗 AI Vehicle Counter (High-Speed Streamlit Version)")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # save video
    tfile = "uploaded_video.mp4"
    with open(tfile, "wb") as f:
        f.write(uploaded_file.read())

    init_excel()
    detector = Detector()
    tracker = CentroidTracker()

    cap = cv2.VideoCapture(tfile)
    ret, frame = cap.read()

    if not ret:
        st.error("Can't read video.")
        st.stop()

    h, w = frame.shape[:2]
    line_y = (h // 2) + LINE_OFFSET

    # streamlit placeholders
    st_frame = st.empty()
    st_up = st.empty()
    st_down = st.empty()

    excel_buffer = []
    frame_id = 0

    FRAME_SCALE = 0.60       # reduce resolution = faster FPS
    ENHANCE_RATE = 2         # enhance every N frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # resize = BIG FPS boost
        frame = cv2.resize(frame, None, fx=FRAME_SCALE, fy=FRAME_SCALE)

        # enhance sometimes to reduce load
        if frame_id % ENHANCE_RATE == 0:
            frame = enhance_night(frame)

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        # counting logic
        for tr in tracks:
            tid = tr["id"]
            hist = tracker.tracks[tid]["history"]

            if len(hist) < MIN_TRACKED_FRAMES:
                continue
            if tracker.tracks[tid]["counted"]:
                continue

            ys = [p[1] for p in hist[-MIN_TRACKED_FRAMES:]]
            if min(ys) < line_y < max(ys):
                direction = "DOWN" if hist[-1][1] > hist[0][1] else "UP"
                if tracker.mark_counted(tid, direction):
                    excel_buffer.append([tid, direction])

        # draw everything
        cv2.line(frame, (0, line_y), (w, line_y), (0, 0, 255), 2)

        for tr in tracks:
            x1, y1, x2, y2 = tr["bbox"]
            cx, cy = tr["centroid"]
            tid = tr["id"]

            color = (0, 255, 0) if not tracker.tracks[tid]["counted"] else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # update UI
        st_up.metric("UP", tracker.up_total)
        st_down.metric("DOWN", tracker.down_total)

        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

    # write Excel once (faster)
    if excel_buffer:
        log_buffer_to_excel(excel_buffer)

    st.success("Video processing completed ✔")
