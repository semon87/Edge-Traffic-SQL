"""
evaluate.py
Academic Evaluation Runner for Edge Text-to-SQL Performance.

Computes:
1. Valid SQL Rate (VSR %): Percentage of queries that compile and execute cleanly.
2. Execution Accuracy (EX %): Ground-truth result set equivalence.
3. Latency & Tokens/sec: Profiling data on NVIDIA Jetson Orin Nano / local CPU.
Exports publishable CSV & JSON reports for conference paper submissions.
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, Any, List

# Ensure parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from track2_query_engine.sql_engine import SafeSqlEngine
from track2_query_engine.llm_client import EdgeLLMClient
from track2_query_engine.pipeline import TrafficQueryPipeline


def normalize_val(v: Any) -> Any:
    """Normalizes values for robust execution accuracy comparisons."""
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def are_results_equal(rows_gen: List[List], rows_gt: List[List]) -> bool:
    """Compares generated SQL rows against ground-truth rows order-independently."""
    if len(rows_gen) != len(rows_gt):
        return False

    # Normalize rows
    norm_gen = [tuple(normalize_val(x) for x in r) for r in rows_gen]
    norm_gt = [tuple(normalize_val(x) for x in r) for r in rows_gt]

    # For scalar single-value results, compare directly
    if len(norm_gen) == 1 and len(norm_gen[0]) == 1:
        return norm_gen[0][0] == norm_gt[0][0]

    try:
        # Compare as multiset / sorted lists
        return sorted(norm_gen) == sorted(norm_gt)
    except TypeError:
        return norm_gen == norm_gt


def run_evaluation(
    benchmark_file: str,
    db_path: str,
    provider: str = "ollama",
    model_name: str = "qwen2.5-coder:3b",
    endpoint: str = "http://localhost:11434",
    output_dir: str = "benchmark_reports"
) -> Dict[str, Any]:
    """Runs the full evaluation benchmark and produces metric reports."""
    os.makedirs(output_dir, exist_ok=True)

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    sql_engine = SafeSqlEngine(db_path)
    llm_client = EdgeLLMClient(
        provider=provider,
        model_name=model_name,
        endpoint=endpoint,
        context_size=2048,
        temperature=0.0
    )
    pipeline = TrafficQueryPipeline(db_path, llm_client=llm_client, enable_self_correction=True)

    print(f"\n========================================================")
    print(f"  Edge Text-to-SQL Evaluation Benchmark")
    print(f"  Target Model: {model_name} (Provider: {provider})")
    print(f"  Database:     {db_path}")
    print(f"  Test Queries: {len(benchmarks)}")
    print(f"========================================================\n")

    results = []
    total_valid_sql = 0
    total_correct_exec = 0
    latencies = []

    by_difficulty = {
        "Simple": {"total": 0, "valid": 0, "correct": 0},
        "Intermediate": {"total": 0, "valid": 0, "correct": 0},
        "Advanced": {"total": 0, "valid": 0, "correct": 0}
    }

    for idx, item in enumerate(benchmarks, 1):
        q_id = item["id"]
        diff = item["difficulty"]
        question = item["question"]
        gt_sql = item["ground_truth_sql"]

        by_difficulty[diff]["total"] += 1

        # Execute ground truth SQL
        gt_res = sql_engine.execute(gt_sql)
        if not gt_res["success"]:
            print(f"[WARN] Ground truth SQL for {q_id} failed: {gt_res['error']}")

        # Run pipeline
        out = pipeline.query(question)
        gen_sql = out["sql"]
        is_valid = out["success"]
        latency = out["llm_latency_seconds"]
        latencies.append(latency)

        if is_valid:
            total_valid_sql += 1
            by_difficulty[diff]["valid"] += 1

        # Check Execution Accuracy
        gen_rows = out.get("execution_result", {}).get("rows", []) if is_valid else []
        gt_rows = gt_res.get("rows", [])
        is_correct = is_valid and are_results_equal(gen_rows, gt_rows)

        if is_correct:
            total_correct_exec += 1
            by_difficulty[diff]["correct"] += 1

        status_str = "PASS" if is_correct else ("VALID_DIFF_RES" if is_valid else "FAIL")
        print(f"[{idx:02d}/{len(benchmarks):02d}] {q_id} ({diff:12s}): {status_str} | Latency: {latency:.2f}s | {question[:45]}...")

        results.append({
            "id": q_id,
            "difficulty": diff,
            "question": question,
            "ground_truth_sql": gt_sql,
            "generated_sql": gen_sql,
            "is_valid_sql": is_valid,
            "is_execution_correct": is_correct,
            "was_self_corrected": out.get("was_corrected", False),
            "latency_seconds": latency,
            "error": out.get("error")
        })

    total_q = len(benchmarks)
    vsr = (total_valid_sql / total_q) * 100.0 if total_q > 0 else 0.0
    ex = (total_correct_exec / total_q) * 100.0 if total_q > 0 else 0.0
    mean_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    summary = {
        "model_name": model_name,
        "provider": provider,
        "total_queries": total_q,
        "valid_sql_rate_percent": round(vsr, 2),
        "execution_accuracy_percent": round(ex, 2),
        "mean_latency_seconds": round(mean_lat, 3),
        "by_difficulty": {
            k: {
                "total": v["total"],
                "valid_sql_rate_percent": round((v["valid"] / v["total"] * 100.0), 2) if v["total"] > 0 else 0.0,
                "execution_accuracy_percent": round((v["correct"] / v["total"] * 100.0), 2) if v["total"] > 0 else 0.0
            }
            for k, v in by_difficulty.items()
        }
    }

    # Print summary table
    print(f"\n========================================================")
    print(f"                   EVALUATION SUMMARY")
    print(f"========================================================")
    print(f"Valid SQL Rate (VSR):       {vsr:.2f}% ({total_valid_sql}/{total_q})")
    print(f"Execution Accuracy (EX):     {ex:.2f}% ({total_correct_exec}/{total_q})")
    print(f"Mean Generation Latency:     {mean_lat:.3f} s")
    print(f"--------------------------------------------------------")
    for diff, stats in summary["by_difficulty"].items():
        print(f"  {diff:12s} - VSR: {stats['valid_sql_rate_percent']:.1f}% | EX: {stats['execution_accuracy_percent']:.1f}% (N={stats['total']})")
    print(f"========================================================\n")

    # Save reports with model-specific names to prevent overwriting
    clean_model = model_name.replace(":", "_").replace("/", "_")
    json_path = os.path.join(output_dir, f"benchmark_summary_{clean_model}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(output_dir, f"benchmark_results_{clean_model}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("id,difficulty,is_valid_sql,is_execution_correct,was_self_corrected,latency_seconds,question,generated_sql\n")
        for r in results:
            clean_q = r["question"].replace('"', '""')
            clean_sql = r["generated_sql"].replace('"', '""').replace("\n", " ")
            f.write(f'"{r["id"]}","{r["difficulty"]}",{r["is_valid_sql"]},{r["is_execution_correct"]},{r["was_self_corrected"]},{r["latency_seconds"]},"{clean_q}","{clean_sql}"\n')

    # Also save latest generic copy
    latest_json = os.path.join(output_dir, "benchmark_summary.json")
    latest_csv = os.path.join(output_dir, "benchmark_results.csv")
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(latest_csv, "w", encoding="utf-8") as f:
        f.write("id,difficulty,is_valid_sql,is_execution_correct,was_self_corrected,latency_seconds,question,generated_sql\n")
        for r in results:
            clean_q = r["question"].replace('"', '""')
            clean_sql = r["generated_sql"].replace('"', '""').replace("\n", " ")
            f.write(f'"{r["id"]}","{r["difficulty"]}",{r["is_valid_sql"]},{r["is_execution_correct"]},{r["was_self_corrected"]},{r["latency_seconds"]},"{clean_q}","{clean_sql}"\n')

    print(f"Reports saved to:\n  - {json_path}\n  - {csv_path}\n")
    return summary


def verify_ground_truth_queries(benchmark_file: str, db_path: str) -> None:
    """Verifies that all ground truth queries in the benchmark file execute with 0 errors."""
    sql_engine = SafeSqlEngine(db_path)
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    print(f"\nVerifying {len(benchmarks)} ground-truth queries against: {db_path}")
    errors = 0
    for q in benchmarks:
        res = sql_engine.execute(q["ground_truth_sql"])
        if not res["success"]:
            print(f"[FAIL] {q['id']}: {res['error']}")
            errors += 1

    if errors == 0:
        print(f"SUCCESS: All {len(benchmarks)} ground-truth SQL queries executed with 0 errors!\n")
    else:
        print(f"Found {errors} errors in ground-truth queries.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Edge Text-to-SQL Pipeline")
    parser.add_argument("--db", default=os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite"))
    parser.add_argument("--benchmarks", default=os.path.join("track2_query_engine", "benchmark_queries.json"))
    parser.add_argument("--provider", default="mock", choices=["ollama", "openai", "mock"])
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--output-dir", default="benchmark_reports", help="Directory to save JSON and CSV reports")
    parser.add_argument("--verify-gt", action="store_true", help="Verify all ground truth queries in SQLite")

    args = parser.parse_args()

    if args.verify_gt:
        verify_ground_truth_queries(args.benchmarks, args.db)
    else:
        run_evaluation(
            benchmark_file=args.benchmarks,
            db_path=args.db,
            provider=args.provider,
            model_name=args.model,
            endpoint=args.endpoint,
            output_dir=args.output_dir
        )

