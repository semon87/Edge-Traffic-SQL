"""
track1_video_pipeline.py
Track 1: Real-Time Edge Video Analytics Pipeline for NVIDIA Jetson Orin Nano.

Performs:
1. Video / Camera frame ingestion (MP4, RTSP, or USB webcam)
2. Vehicle Detection using YOLO11n
3. Multi-Object Tracking using ByteTrack
4. Speed estimation & kinematic trajectory analysis
5. Real-time batched persistence into traffic_analytics.sqlite (WAL mode)
"""

import os
import sys
import time
import math
import sqlite3
import argparse
from datetime import datetime, timedelta

# Default Database Path
DEFAULT_DB = os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite")
SPEED_LIMIT_KMH = 50.0


def init_database(db_path: str):
    """Initializes SQLite database with WAL mode and required tables."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Enable WAL mode for zero lock contention with Track 2
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")
    cur.execute("PRAGMA busy_timeout = 5000;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sequences (
      sequence_id TEXT PRIMARY KEY,
      video_filename TEXT NOT NULL,
      fps REAL CHECK (fps > 0),
      frame_width INTEGER CHECK (frame_width > 0),
      frame_height INTEGER CHECK (frame_height > 0),
      total_frames INTEGER CHECK (total_frames >= 0),
      processing_timestamp TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_tracks (
      id INTEGER PRIMARY KEY,
      sequence_id TEXT NOT NULL,
      track_id INTEGER NOT NULL CHECK (track_id >= 0),
      vehicle_type TEXT NOT NULL,
      vehicle_colour TEXT NOT NULL,
      movement_direction TEXT NOT NULL,
      first_frame INTEGER NOT NULL CHECK (first_frame >= 0),
      last_frame INTEGER NOT NULL CHECK (last_frame >= first_frame),
      entry_time_seconds REAL NOT NULL,
      exit_time_seconds REAL NOT NULL,
      observed_duration_seconds REAL NOT NULL,
      number_of_observations INTEGER NOT NULL,
      entry_datetime TEXT,
      exit_datetime TEXT,
      avg_speed_kmh REAL DEFAULT 0.0,
      is_speeding INTEGER DEFAULT 0,
      UNIQUE(sequence_id, track_id),
      FOREIGN KEY(sequence_id) REFERENCES sequences(sequence_id) ON DELETE CASCADE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_observations (
      id INTEGER PRIMARY KEY,
      sequence_id TEXT NOT NULL,
      track_id INTEGER NOT NULL,
      frame_index INTEGER NOT NULL CHECK (frame_index >= 0),
      timestamp_seconds REAL NOT NULL,
      bbox_x REAL NOT NULL,
      bbox_y REAL NOT NULL,
      bbox_width REAL NOT NULL,
      bbox_height REAL NOT NULL,
      detection_confidence REAL NOT NULL,
      UNIQUE(sequence_id, track_id, frame_index),
      FOREIGN KEY(sequence_id, track_id) REFERENCES vehicle_tracks(sequence_id, track_id) ON DELETE CASCADE
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_sequence ON vehicle_tracks(sequence_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_datetime ON vehicle_tracks(entry_datetime);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tracks_speed ON vehicle_tracks(avg_speed_kmh);")
    conn.commit()
    conn.close()


def determine_direction(dx: float, dy: float, min_disp: float = 15.0) -> str:
    """Determines principal motion direction from trajectory displacement."""
    dist = math.hypot(dx, dy)
    if dist < min_disp:
        return "unknown"
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    else:
        return "up" if dy < 0 else "down"


def estimate_speed(u1: float, v1: float, u2: float, v2: float, duration_seconds: float) -> float:
    """Estimates speed in km/h using depth-aware perspective scaling."""
    if duration_seconds < 0.25:
        return 0.0
    dist_px = math.hypot(u2 - u1, v2 - v1)
    if dist_px < 15.0:
        return 0.0
    mid_v = (v1 + v2) / 2.0
    scale = 0.045 + 0.165 * max(0.0, min(1.0, (540.0 - mid_v) / 540.0))
    speed_kmh = (dist_px * scale / duration_seconds) * 3.6
    return round(max(0.0, min(110.0, speed_kmh)), 1)


def run_pipeline(
    video_source: str,
    sequence_id: str = "live_stream",
    model_path: str = "yolo11n.pt",
    db_path: str = DEFAULT_DB,
    confidence: float = 0.10,
    device: str = "0",
    max_frames: int = 0,
    min_hits: int = 5
):
    """Runs YOLO11n + ByteTrack and logs records into traffic_analytics.sqlite.
    Note: confidence threshold defaults to 0.10 to feed ByteTrack's two-stage association
    (high tier >= 0.35 and low tier 0.10 <= conf < 0.35).
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        print("Error: Missing packages. Run: pip install ultralytics opencv-python")
        sys.exit(1)

    print(f"[Track 1] Initializing SQLite database at: {db_path}")
    init_database(db_path)

    print(f"[Track 1] Loading YOLO model: {model_path} on device {device}...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_source if not video_source.isdigit() else int(video_source))
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 960
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    base_time = datetime.now()

    conn = sqlite3.connect(db_path, timeout=10.0)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO sequences (sequence_id, video_filename, fps, frame_width, frame_height, total_frames, processing_timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sequence_id, str(video_source), fps, width, height, total_video_frames, base_time.isoformat()))
    conn.commit()

    active_tracks = {}
    frame_idx = 0
    start_time = time.time()

    print(f"[Track 1] Starting video tracking on {sequence_id} (FPS: {fps}, Res: {width}x{height})...")

    # Map COCO classes to vehicle types
    COCO_VEHICLE_MAP = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames > 0 and frame_idx >= max_frames):
                break

            timestamp_seconds = frame_idx / fps
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=confidence,
                device=device,
                verbose=False
            )

            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confidences):
                    if cls_id not in COCO_VEHICLE_MAP:
                        continue  # Skip non-vehicle detections

                    v_type = COCO_VEHICLE_MAP[cls_id]
                    bx, by, bw, bh = box
                    # Top-left representation
                    tl_x = bx - bw / 2.0
                    tl_y = by - bh / 2.0
                    u = bx
                    v = by + bh / 2.0

                    if track_id not in active_tracks:
                        active_tracks[track_id] = {
                            "type": v_type,
                            "colour": "grey",
                            "first_frame": frame_idx,
                            "first_u": u,
                            "first_v": v,
                            "last_frame": frame_idx,
                            "last_u": u,
                            "last_v": v,
                            "obs_count": 0
                        }

                    t = active_tracks[track_id]
                    t["last_frame"] = frame_idx
                    t["last_u"] = u
                    t["last_v"] = v
                    t["obs_count"] += 1

                    # Log observation
                    cur.execute("""
                        INSERT OR IGNORE INTO vehicle_observations
                        (sequence_id, track_id, frame_index, timestamp_seconds, bbox_x, bbox_y, bbox_width, bbox_height, detection_confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (sequence_id, int(track_id), frame_idx, timestamp_seconds, float(tl_x), float(tl_y), float(bw), float(bh), float(conf)))

            frame_idx += 1
            if frame_idx % 15 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[Track 1] Processed {frame_idx} frames | Processing Speed: {fps_proc:.1f} FPS | Active Tracks: {len(active_tracks)}")
                conn.commit()

    finally:
        cap.release()
        saved_tracks = 0
        print(f"[Track 1] Filtering and committing tracks (min_hits={min_hits})...")

        for track_id, t in active_tracks.items():
            if t["obs_count"] < min_hits:
                continue  # Filter out single-frame / short-lived transient noise

            saved_tracks += 1
            first_f = t["first_frame"]
            last_f = t["last_frame"]
            entry_s = first_f / fps
            exit_s = last_f / fps
            dur_s = exit_s - entry_s

            entry_dt = (base_time + timedelta(seconds=entry_s)).strftime("%Y-%m-%d %H:%M:%S")
            exit_dt = (base_time + timedelta(seconds=exit_s)).strftime("%Y-%m-%d %H:%M:%S")

            dx = t["last_u"] - t["first_u"]
            dy = t["last_v"] - t["first_v"]
            direction = determine_direction(dx, dy)
            speed = estimate_speed(t["first_u"], t["first_v"], t["last_u"], t["last_v"], dur_s)
            is_speeding = 1 if speed > SPEED_LIMIT_KMH else 0

            cur.execute("""
                INSERT OR REPLACE INTO vehicle_tracks
                (sequence_id, track_id, vehicle_type, vehicle_colour, movement_direction,
                 first_frame, last_frame, entry_time_seconds, exit_time_seconds,
                 observed_duration_seconds, number_of_observations, entry_datetime, exit_datetime, avg_speed_kmh, is_speeding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (sequence_id, track_id, t["type"], t["colour"], direction,
                  first_f, last_f, entry_s, exit_s, dur_s, t["obs_count"],
                  entry_dt, exit_dt, speed, is_speeding))

        conn.commit()
        conn.close()
        total_time = time.time() - start_time
        print(f"[Track 1] Complete! Processed {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} FPS). Committed {saved_tracks} verified tracks (filtered {len(active_tracks) - saved_tracks} noise tracks).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track 1: Video Analytics Edge Pipeline")
    parser.add_argument("--source", "-s", required=True, help="Path to video file (.mp4), folder, or camera index (e.g. 0)")
    parser.add_argument("--sequence", default="demo_seq", help="Sequence identifier")
    parser.add_argument("--model", default="yolo11n.engine", help="YOLO weights or TensorRT engine file")
    parser.add_argument("--db", default=DEFAULT_DB, help="Target SQLite database path")
    parser.add_argument("--conf", type=float, default=0.10, help="Detection floor threshold (0.10 feeds ByteTrack two-tier association)")
    parser.add_argument("--min-hits", type=int, default=5, help="Minimum frame detections required to save track")
    parser.add_argument("--device", default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")

    args = parser.parse_args()
    run_pipeline(
        video_source=args.source,
        sequence_id=args.sequence,
        model_path=args.model,
        db_path=args.db,
        confidence=args.conf,
        device=args.device,
        max_frames=args.max_frames,
        min_hits=args.min_hits
    )

