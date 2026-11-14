"""
sql_engine.py
Safe, Read-Only SQLite Execution Engine with Edge Concurrency Optimizations.

Designed for concurrent execution on edge devices (NVIDIA Jetson Orin Nano)
alongside continuous background video analytics writers.
"""

import sqlite3
import time
import re
from typing import Any, Dict, List, Optional, Tuple


class SafeSqlEngine:
    """Safe read-only execution engine for SQLite databases."""

    # Disallowed mutating SQL keywords
    DISALLOWED_KEYWORDS = (
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|VACUUM|REINDEX)\b"
    )

    def __init__(self, db_path: str, timeout_seconds: float = 3.0):
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        self._ensure_wal_mode()

    def _ensure_wal_mode(self) -> None:
        """
        Ensures Write-Ahead Logging (WAL) is enabled on the database so
        Track 1 background writers do not block Track 2 readers.
        """
        try:
            # Connect in read-write temporarily to ensure WAL mode
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA busy_timeout=5000;")
            conn.commit()
            conn.close()
        except Exception as e:
            # Non-fatal if the file is currently locked or read-only
            print(f"[SafeSqlEngine] Note: Could not set WAL mode ({e}). Proceeding in read-only mode.")

    def _get_read_only_connection(self) -> sqlite3.Connection:
        """Opens a strictly read-only URI connection to prevent any disk mutations."""
        # Use URI mode=ro for guaranteed read-only safety
        uri = f"file:{self.db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
        conn.row_factory = sqlite3.Row
        return conn

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validates that a query is a read-only SELECT or EXPLAIN statement.
        Returns (is_valid, error_message).
        """
        cleaned = query.strip()
        if not cleaned:
            return False, "Query is empty."

        # Strip comments
        cleaned = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL).strip()

        # Check for disallowed mutating keywords
        match = re.search(self.DISALLOWED_KEYWORDS, cleaned, re.IGNORECASE)
        if match:
            return False, f"Mutating query rejected: contains forbidden keyword '{match.group(1).upper()}'."

        # Ensure query starts with SELECT, WITH, or EXPLAIN
        if not re.match(r"^(SELECT|WITH|EXPLAIN)\b", cleaned, re.IGNORECASE):
            return False, "Only SELECT or WITH queries are permitted."

        return True, None

    def execute(self, query: str) -> Dict[str, Any]:
        """
        Executes a SQL query safely, returning columns, rows, execution time, and status.
        """
        is_valid, error = self.validate_query(query)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "execution_time_ms": 0.0,
            }

        start_time = time.perf_counter()
        conn = None
        try:
            conn = self._get_read_only_connection()
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            columns = [desc[0] for desc in cur.description] if cur.description else []
            # Convert sqlite3.Row objects to standard lists for serialization
            serialized_rows = [[row[col] for col in columns] for row in rows]

            return {
                "success": True,
                "error": None,
                "columns": columns,
                "rows": serialized_rows,
                "row_count": len(serialized_rows),
                "execution_time_ms": round(elapsed_ms, 2),
            }
        except sqlite3.Error as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "success": False,
                "error": str(e),
                "columns": [],
                "rows": [],
                "row_count": 0,
                "execution_time_ms": round(elapsed_ms, 2),
            }
        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    import os
    db_test_path = os.path.join("..", "outputs", "jetson_tuned", "traffic_analytics.sqlite")
    if not os.path.exists(db_test_path):
        db_test_path = os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite")

    print(f"Testing SafeSqlEngine with DB at: {db_test_path}")
    engine = SafeSqlEngine(db_test_path)

    # Test 1: Valid SELECT query
    res = engine.execute("SELECT vehicle_type, count(*) as count FROM vehicle_tracks GROUP BY vehicle_type;")
    print("Test 1 Result:", res)

    # Test 2: Mutation rejection
    res2 = engine.execute("DROP TABLE sequences;")
    print("Test 2 Result (Should reject):", res2["error"])

