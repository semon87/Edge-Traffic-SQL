"""
pipeline.py
End-to-End Edge Text-to-SQL Query Pipeline with Self-Correction.

Translates Natural Language Traffic Queries -> SQL -> SQLite Execution -> Answer.
Runs deterministically on edge devices (NVIDIA Jetson Orin Nano).
"""

import time
import os
from typing import Dict, Any, Optional

from track2_query_engine.sql_engine import SafeSqlEngine
from track2_query_engine.schema import SchemaExtractor
from track2_query_engine.prompts import (
    build_text_to_sql_prompt,
    build_correction_prompt,
    synthesize_answer_template
)
from track2_query_engine.llm_client import EdgeLLMClient


class TrafficQueryPipeline:
    """End-to-End Orchestrator for Natural Language Traffic Analytics."""

    def __init__(
        self,
        db_path: str,
        llm_client: Optional[EdgeLLMClient] = None,
        enable_self_correction: bool = True
    ):
        self.db_path = db_path
        self.sql_engine = SafeSqlEngine(db_path)
        self.schema_extractor = SchemaExtractor(db_path)
        self.llm_client = llm_client or EdgeLLMClient(provider="ollama")
        self.enable_self_correction = enable_self_correction

        # Precompute the cached token-efficient prompt schema
        self._cached_schema = self.schema_extractor.get_prompt_schema()

    def query(self, question: str) -> Dict[str, Any]:
        """
        Executes a complete natural language query lifecycle:
        1. Construct prompt
        2. Query edge SLM
        3. Validate & Execute SQL
        4. Self-correct if syntax fails (1 retry)
        5. Synthesize final natural language answer
        """
        start_total = time.perf_counter()
        pipeline_log = []

        # Step 1: Build the prompt
        prompt = build_text_to_sql_prompt(self._cached_schema, question)

        # Step 2: Generate initial SQL
        llm_res = self.llm_client.generate(prompt)
        initial_sql = llm_res.get("sql_query", "").strip()
        llm_latency = llm_res.get("latency_seconds", 0.0)

        if not llm_res.get("success"):
            err_msg = llm_res.get('error')
            return {
                "success": False,
                "question": question,
                "sql": initial_sql,
                "error": f"LLM Generation failed: {err_msg}",
                "answer": f"Unable to generate query due to LLM error: {err_msg}",
                "execution_result": None,
                "was_corrected": False,
                "llm_latency_seconds": llm_latency,
                "total_latency_seconds": round(time.perf_counter() - start_total, 3)
            }

        # Step 3: Execute SQL against local SQLite database
        exec_res = self.sql_engine.execute(initial_sql)
        was_corrected = False
        final_sql = initial_sql

        # Step 4: Self-Correction Loop (if SQL failed and enabled)
        if not exec_res["success"] and self.enable_self_correction:
            pipeline_log.append(f"Initial SQL failed ({exec_res['error']}). Triggering self-correction...")
            correction_prompt = build_correction_prompt(
                self._cached_schema,
                question,
                initial_sql,
                exec_res["error"]
            )
            retry_res = self.llm_client.generate(correction_prompt)
            corrected_sql = retry_res.get("sql_query", "").strip()

            if retry_res.get("success") and corrected_sql:
                retry_exec = self.sql_engine.execute(corrected_sql)
                if retry_exec["success"]:
                    exec_res = retry_exec
                    final_sql = corrected_sql
                    was_corrected = True
                    llm_latency += retry_res.get("latency_seconds", 0.0)
                    pipeline_log.append("Self-correction successful.")

        # Step 5: Synthesize human-readable answer
        if exec_res["success"]:
            answer = synthesize_answer_template(question, exec_res)
        else:
            answer = f"Could not compute an answer. SQL Execution Error: {exec_res['error']}"

        total_elapsed = round(time.perf_counter() - start_total, 3)

        return {
            "success": exec_res["success"],
            "question": question,
            "sql": final_sql,
            "error": exec_res.get("error"),
            "answer": answer,
            "execution_result": exec_res,
            "was_corrected": was_corrected,
            "pipeline_log": pipeline_log,
            "llm_latency_seconds": round(llm_latency, 3),
            "total_latency_seconds": total_elapsed
        }


if __name__ == "__main__":
    test_db = os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite")
    mock_client = EdgeLLMClient(provider="mock")
    pipeline = TrafficQueryPipeline(test_db, llm_client=mock_client)

    test_q = "How many buses were observed?"
    print(f"\n--- Testing Pipeline with Query: '{test_q}' ---")
    out = pipeline.query(test_q)
    print("SQL:", out["sql"])
    print("Answer:", out["answer"])
    print("Total Latency:", out["total_latency_seconds"], "s")

