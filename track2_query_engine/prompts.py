"""
prompts.py
Prompt Templates and Complex Few-Shot Demonstrations for Edge Text-to-SQL.

Calibrated specifically for 1.5B - 3B parameter models (e.g. Qwen2.5-Coder-3B)
to produce clean, deterministic SQLite syntax for complex temporal and kinematic queries.
"""

from typing import List, Dict, Optional


SYSTEM_INSTRUCTION = """You are an edge AI assistant that translates natural language traffic questions into exact SQLite queries.
Given the SQLite schema and the user's question, write ONLY a valid, read-only SQL query inside ```sql ... ```.

STRICT RULES:
1. Generate ONLY SQLite-compatible SELECT or WITH statements. Never write INSERT, UPDATE, DELETE, or DROP.
2. Only use columns and tables defined in the schema.
3. For dates and times:
   - For 'today', use: (date(entry_datetime) = '2026-11-05' OR date(entry_datetime) = CURRENT_DATE)
   - For time ranges like 'between 10-11 am' or '10 AM to 11 AM', use: time(entry_datetime) BETWEEN '10:00:00' AND '11:00:00'
   - For hourly grouping, use: strftime('%H:00', entry_datetime) AS hour
4. For speed & violations:
   - Speed is stored in `avg_speed_kmh` (in km/h).
   - Speeding flag is `is_speeding` (1 if speed > 50.0 km/h, else 0).
5. For categorical fields (vehicle_type, vehicle_colour, movement_direction), use lowercase values from the schema.
6. Provide meaningful column aliases (e.g., COUNT(*) AS total_count, AVG(avg_speed_kmh) AS avg_speed).
7. Output NO pleasantries, markdown text, or explanations outside the ```sql ... ``` block."""


FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {
        "question": "Show all blue buses that moved right today between 10 AM and 11 AM.",
        "sql": """SELECT sequence_id, track_id, vehicle_type, vehicle_colour, movement_direction, avg_speed_kmh, entry_datetime
FROM vehicle_tracks
WHERE vehicle_type = 'bus'
  AND vehicle_colour = 'blue'
  AND movement_direction = 'right'
  AND (date(entry_datetime) = '2026-11-05' OR date(entry_datetime) = CURRENT_DATE)
  AND time(entry_datetime) BETWEEN '10:00:00' AND '11:00:00'
ORDER BY entry_datetime ASC;"""
    },
    {
        "question": "List video filenames that recorded an average vehicle speed greater than 40 km/h.",
        "sql": """SELECT s.video_filename, AVG(t.avg_speed_kmh) AS overall_avg_speed
FROM sequences s
JOIN vehicle_tracks t ON s.sequence_id = t.sequence_id
GROUP BY s.video_filename
HAVING overall_avg_speed > 40.0
ORDER BY overall_avg_speed DESC;"""
    },
    {
        "question": "Find the 5 vehicle tracks with the highest number of observations.",
        "sql": """SELECT sequence_id, track_id, vehicle_type, number_of_observations, observed_duration_seconds
FROM vehicle_tracks
ORDER BY number_of_observations DESC
LIMIT 5;"""
    },
    {
        "question": "Count the number of speeding violations that occurred during each hour today.",
        "sql": """SELECT strftime('%H:00', entry_datetime) AS hour_window, COUNT(*) AS violation_count
FROM vehicle_tracks
WHERE is_speeding = 1
  AND (date(entry_datetime) = '2026-11-05' OR date(entry_datetime) = CURRENT_DATE)
GROUP BY hour_window
ORDER BY hour_window ASC;"""
    },
    {
        "question": "Which vehicle colors have an average speed above 35 km/h across at least 10 recorded tracks?",
        "sql": """SELECT vehicle_colour, COUNT(*) AS vehicle_count, AVG(avg_speed_kmh) AS avg_speed
FROM vehicle_tracks
GROUP BY vehicle_colour
HAVING vehicle_count >= 10 AND avg_speed > 35.0
ORDER BY avg_speed DESC;"""
    }
]


def build_text_to_sql_prompt(schema_str: str, user_question: str) -> str:
    """Builds the full Text-to-SQL prompt including schema and complex few-shot pairs."""
    prompt_parts = [
        SYSTEM_INSTRUCTION,
        "\n" + schema_str,
        "\n### Demonstrations:"
    ]

    for ex in FEW_SHOT_EXAMPLES:
        prompt_parts.append(f"\nQuestion: {ex['question']}\n```sql\n{ex['sql']}\n```")

    prompt_parts.append(f"\n### Current Task:\nQuestion: {user_question}\n```sql")
    return "\n".join(prompt_parts)


def build_correction_prompt(
    schema_str: str,
    user_question: str,
    failed_sql: str,
    error_message: str
) -> str:
    """
    Constructs a feedback prompt for the self-correction loop when
    SQLite reports a syntax error or missing column.
    """
    return f"""{SYSTEM_INSTRUCTION}

{schema_str}

The following SQL query was generated for the question but failed execution:
Question: {user_question}
Failed SQL:
```sql
{failed_sql}
```
SQLite Execution Error: {error_message}

Please fix the error and provide the corrected, valid SQLite query.
Output ONLY the corrected query inside ```sql ... ```:"""


def synthesize_answer_template(user_question: str, query_result: Dict) -> str:
    """
    Deterministic rule-based synthesizer that turns complex SQL results into
    human-readable answers without requiring a second LLM inference pass.
    Saves edge GPU compute and memory on Jetson Orin Nano.
    """
    if not query_result.get("success"):
        return f"Error executing query: {query_result.get('error')}"

    columns = query_result.get("columns", [])
    rows = query_result.get("rows", [])
    row_count = query_result.get("row_count", 0)

    if row_count == 0:
        return "No matching records were found in the traffic database."

    # Single scalar result (e.g. COUNT(*), AVG(speed))
    if row_count == 1 and len(columns) == 1:
        val = rows[0][0]
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
        col_name = columns[0].replace("_", " ")
        return f"The result for '{col_name}' is **{val_str}**."

    # Single row multiple attributes
    if row_count == 1:
        items = [f"**{col.replace('_', ' ')}**: {val:.2f}" if isinstance(val, float) else f"**{col.replace('_', ' ')}**: {val}" for col, val in zip(columns, rows[0])]
        return "Found 1 matching record:\n* " + "\n* ".join(items)

    # Moderate table (<= 5 rows)
    if row_count <= 5:
        summary_lines = [f"Found {row_count} records:"]
        for row in rows:
            line = ", ".join(f"{col}: {val:.2f}" if isinstance(val, float) else f"{col}: {val}" for col, val in zip(columns, row))
            summary_lines.append(f"- {line}")
        return "\n".join(summary_lines)

    # Larger table (> 5 rows)
    sample_preview = ", ".join(f"{col}: {val}" for col, val in zip(columns, rows[0]))
    return f"Retrieved **{row_count} matching vehicles**. (First record: {sample_preview})"
