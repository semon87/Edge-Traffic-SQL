"""
llm_client.py
Zero-Dependency Edge LLM Client for Ollama & OpenAI-Compatible Servers.

Uses Python standard library (urllib.request, json) to avoid pip dependency issues
on both development laptops and target Jetson Orin Nano devices.
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, Tuple


class EdgeLLMClient:
    """Client for edge-based LLM inference (Ollama or llama-server)."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "qwen2.5-coder:3b",
        endpoint: str = "http://localhost:11434",
        context_size: int = 2048,
        temperature: float = 0.0,
        timeout: float = 120.0
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.endpoint = endpoint.rstrip("/")
        self.context_size = context_size
        self.temperature = temperature
        self.timeout = timeout

    def check_health(self) -> Tuple[bool, str]:
        """Checks if the local LLM server is up and responsive."""
        if self.provider == "mock":
            return True, "Mock engine active (offline mode)."

        try:
            req = urllib.request.Request(self.endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status in (200, 404):  # 404 is standard for root on some servers
                    return True, f"Server running at {self.endpoint}"
        except urllib.error.URLError as e:
            return False, f"Cannot connect to {self.endpoint}: {e.reason}"
        except Exception as e:
            return False, f"Connection error: {e}"

        return True, "Server reachable."

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Dispatches prompt to Ollama, OpenAI-compatible server, or Mock.
        Returns dict with: raw_response, sql_query, latency_seconds, tokens_per_second.
        """
        if self.provider == "mock":
            return self._generate_mock(prompt)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt)
        elif self.provider == "openai":
            return self._generate_openai_compatible(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_ollama(self, prompt: str) -> Dict[str, Any]:
        """Calls Ollama's /api/generate endpoint."""
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "24h",  # Keep resident in Jetson RAM to avoid repeated reload delays
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_size,  # Strict memory cap for Jetson Orin Nano
                "stop": ["```\n\n", "Question:"]
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - start_time

            raw_text = result.get("response", "")
            eval_count = result.get("eval_count", 0)
            eval_duration_ns = result.get("eval_duration", 0)
            tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

            sql = self.extract_sql(raw_text)

            return {
                "success": True,
                "raw_response": raw_text,
                "sql_query": sql,
                "latency_seconds": round(elapsed, 3),
                "eval_tokens": eval_count,
                "tokens_per_second": round(tps, 1),
                "error": None
            }
        except urllib.error.HTTPError as e:
            elapsed = time.perf_counter() - start_time
            err_body = e.read().decode("utf-8") if e.fp else str(e)
            print(f"[EdgeLLMClient HTTP {e.code} Error]: {err_body}")
            return {
                "success": False,
                "raw_response": "",
                "sql_query": "",
                "latency_seconds": round(elapsed, 3),
                "eval_tokens": 0,
                "tokens_per_second": 0.0,
                "error": f"HTTP {e.code}: {err_body}"
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"[EdgeLLMClient Error]: {e}")
            return {
                "success": False,
                "raw_response": "",
                "sql_query": "",
                "latency_seconds": round(elapsed, 3),
                "eval_tokens": 0,
                "tokens_per_second": 0.0,
                "error": str(e)
            }

    def _generate_openai_compatible(self, prompt: str) -> Dict[str, Any]:
        """Calls standard OpenAI /v1/chat/completions endpoint (e.g. llama-server)."""
        url = f"{self.endpoint}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 256
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            elapsed = time.perf_counter() - start_time

            raw_text = result["choices"][0]["message"]["content"]
            sql = self.extract_sql(raw_text)

            return {
                "success": True,
                "raw_response": raw_text,
                "sql_query": sql,
                "latency_seconds": round(elapsed, 3),
                "eval_tokens": result.get("usage", {}).get("completion_tokens", 0),
                "tokens_per_second": 0.0,
                "error": None
            }
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            return {
                "success": False,
                "raw_response": "",
                "sql_query": "",
                "latency_seconds": round(elapsed, 3),
                "eval_tokens": 0,
                "tokens_per_second": 0.0,
                "error": str(e)
            }

    def _generate_mock(self, prompt: str) -> Dict[str, Any]:
        """Offline heuristic mock generator for local testing without an active daemon."""
        time.sleep(0.05)
        # Extract the question from the prompt
        match = re.search(r"Question:\s*(.*?)(?:\n|```|$)", prompt.split("### Current Task:")[-1])
        q = match.group(1).lower() if match else prompt.lower()

        sql = "SELECT COUNT(*) FROM vehicle_tracks;"
        if "red" in q and ("left" in q or "3-4" in q or "3 pm" in q):
            sql = """SELECT sequence_id, track_id, vehicle_type, vehicle_colour, movement_direction, avg_speed_kmh, entry_datetime
FROM vehicle_tracks
WHERE vehicle_colour = 'red'
  AND movement_direction = 'left'
  AND (date(entry_datetime) = '2026-11-05' OR date(entry_datetime) = CURRENT_DATE)
  AND time(entry_datetime) BETWEEN '15:00:00' AND '16:00:00'
ORDER BY entry_datetime ASC;"""
        elif "speeding" in q:
            sql = """SELECT vehicle_type, COUNT(*) AS speeding_count, AVG(avg_speed_kmh) AS avg_speed
FROM vehicle_tracks
WHERE is_speeding = 1
GROUP BY vehicle_type
ORDER BY speeding_count DESC;"""
        elif "fastest" in q:
            sql = """SELECT sequence_id, track_id, vehicle_type, vehicle_colour, avg_speed_kmh, entry_datetime
FROM vehicle_tracks
ORDER BY avg_speed_kmh DESC
LIMIT 3;"""
        elif "hour" in q:
            sql = """SELECT strftime('%H:00', entry_datetime) AS hour, COUNT(*) AS vehicle_count
FROM vehicle_tracks
WHERE (date(entry_datetime) = '2026-11-05' OR date(entry_datetime) = CURRENT_DATE)
GROUP BY hour
ORDER BY vehicle_count DESC;"""
        elif "bus" in q:
            sql = "SELECT COUNT(*) AS bus_count FROM vehicle_tracks WHERE vehicle_type = 'bus';"
        elif "truck" in q:
            sql = "SELECT COUNT(*) AS truck_count FROM vehicle_tracks WHERE vehicle_type = 'truck';"
        elif "car" in q and "white" in q:
            sql = "SELECT COUNT(*) AS white_cars FROM vehicle_tracks WHERE vehicle_type = 'car' AND vehicle_colour = 'white';"
        elif "duration" in q or "dwell" in q:
            sql = "SELECT AVG(observed_duration_seconds) AS avg_duration FROM vehicle_tracks;"
        elif "direction" in q:
            sql = "SELECT movement_direction, COUNT(*) AS count FROM vehicle_tracks WHERE movement_direction != 'unknown' GROUP BY movement_direction;"

        return {
            "success": True,
            "raw_response": f"```sql\n{sql}\n```",
            "sql_query": sql,
            "latency_seconds": 0.05,
            "eval_tokens": 25,
            "tokens_per_second": 500.0,
            "error": None
        }

    @staticmethod
    def extract_sql(text: str) -> str:
        """Extracts clean SQL code from markdown code blocks or plain text."""
        # 1. Look for ```sql ... ```
        sql_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_match:
            candidate = sql_match.group(1).strip()
            # Remove any trailing markdown artifacts
            return candidate.rstrip(";") + ";"

        # 2. Look for any SELECT ... ; statement
        select_match = re.search(r"(SELECT\b.*?;)", text, re.DOTALL | re.IGNORECASE)
        if select_match:
            return select_match.group(1).strip()

        # 3. Fallback: return cleaned text
        return text.strip().rstrip(";") + ";"

