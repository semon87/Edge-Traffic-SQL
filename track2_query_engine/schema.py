"""
schema.py
Token-Efficient Database Schema Extractor & Categorical Value Cache.

Extracts minimal DDL, temporal columns, speed metrics, and valid categorical enums
from traffic_analytics.sqlite to ensure prompt brevity (< 500 tokens) on edge models.
"""

import sqlite3
from typing import Dict, List, Any


class SchemaExtractor:
    """Inspects the SQLite database to produce token-efficient prompt schemas."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._cache = {}

    def get_categorical_enums(self) -> Dict[str, List[str]]:
        """Extracts unique categorical values from the database."""
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cur = conn.cursor()

        enums = {}
        try:
            # Distinct vehicle types
            cur.execute("SELECT DISTINCT vehicle_type FROM vehicle_tracks WHERE vehicle_type IS NOT NULL ORDER BY vehicle_type;")
            enums["vehicle_type"] = [r[0] for r in cur.fetchall()]

            # Distinct vehicle colours
            cur.execute("SELECT DISTINCT vehicle_colour FROM vehicle_tracks WHERE vehicle_colour IS NOT NULL ORDER BY vehicle_colour;")
            enums["vehicle_colour"] = [r[0] for r in cur.fetchall()]

            # Distinct movement directions
            cur.execute("SELECT DISTINCT movement_direction FROM vehicle_tracks WHERE movement_direction IS NOT NULL ORDER BY movement_direction;")
            enums["movement_direction"] = [r[0] for r in cur.fetchall()]

            # Available sequences
            cur.execute("SELECT sequence_id FROM sequences ORDER BY sequence_id;")
            enums["sequence_id"] = [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

        return enums

    def get_prompt_schema(self, include_observations: bool = False) -> str:
        """
        Generates a concise, structured markdown DDL schema for the LLM prompt.
        Includes temporal datetime functions and kinematic speed columns.
        """
        enums = self.get_categorical_enums()

        v_types = ", ".join(f"'{v}'" for v in enums.get("vehicle_type", []))
        v_colors = ", ".join(f"'{c}'" for c in enums.get("vehicle_colour", []))
        v_dirs = ", ".join(f"'{d}'" for d in enums.get("movement_direction", []))

        schema = f"""### SQLite Database Schema

TABLE sequences (
  sequence_id TEXT PRIMARY KEY, -- e.g., 'detrac_1_MVI_20011', 'detrac_10_MVI_20062'
  fps REAL,                     -- video framerate (25.0)
  frame_width INTEGER,          -- video width (960)
  frame_height INTEGER,         -- video height (540)
  total_frames INTEGER          -- total frames in sequence
);

TABLE vehicle_tracks (
  id INTEGER PRIMARY KEY,
  sequence_id TEXT,             -- Foreign key to sequences(sequence_id)
  track_id INTEGER,             -- Unique tracker ID within the sequence
  vehicle_type TEXT,            -- Allowed values: [{v_types}]
  vehicle_colour TEXT,          -- Allowed values: [{v_colors}]
  movement_direction TEXT,      -- Allowed values: [{v_dirs}]
  observed_duration_seconds REAL, -- Dwell time in seconds
  number_of_observations INTEGER, -- Persistence (frame count)
  entry_datetime TEXT,          -- ISO timestamp 'YYYY-MM-DD HH:MM:SS' (e.g. '2026-11-05 15:10:00')
  exit_datetime TEXT,           -- ISO timestamp 'YYYY-MM-DD HH:MM:SS'
  avg_speed_kmh REAL,           -- Estimated vehicle speed in km/h
  is_speeding INTEGER,          -- 1 if avg_speed_kmh > 50.0 km/h, else 0
  FOREIGN KEY(sequence_id) REFERENCES sequences(sequence_id)
);

-- SQLite Date/Time Helper Functions:
-- date(entry_datetime) = '2026-11-05' or CURRENT_DATE
-- time(entry_datetime) BETWEEN '15:00:00' AND '16:00:00' (for 3-4 PM)
-- strftime('%H', entry_datetime) = '15' (hour extraction)
"""

        if include_observations:
            schema += """
TABLE vehicle_observations (
  id INTEGER PRIMARY KEY,
  sequence_id TEXT,
  track_id INTEGER,
  frame_index INTEGER,
  bbox_x REAL, bbox_y REAL, bbox_width REAL, bbox_height REAL,
  detection_confidence REAL,
  FOREIGN KEY(sequence_id, track_id) REFERENCES vehicle_tracks(sequence_id, track_id)
);
"""
        return schema.strip()


if __name__ == "__main__":
    import os
    db_test_path = os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite")
    extractor = SchemaExtractor(db_test_path)
    print("--- Enums ---")
    print(extractor.get_categorical_enums())
    print("\n--- Prompt Schema ---")
    print(extractor.get_prompt_schema())
