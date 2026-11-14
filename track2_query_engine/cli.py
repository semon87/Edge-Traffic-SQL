"""
cli.py
Interactive Command-Line Interface for Edge Traffic Query Engine.

Allows live natural language queries against traffic_analytics.sqlite
with options to inspect generated SQL, query latency, and raw result sets.
"""

import os
import sys
import argparse
from typing import Optional

# Ensure parent directory is in sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from track2_query_engine.llm_client import EdgeLLMClient
from track2_query_engine.pipeline import TrafficQueryPipeline


def print_banner(model: str, provider: str, db_path: str):
    print("=" * 65)
    print("  Edge Traffic Query Engine (Track 2: NL -> Text-to-SQL)")
    print(f"  Model:    {model} (via {provider})")
    print(f"  Database: {db_path}")
    print("=" * 65)
    print("Type your traffic question in natural language (or 'exit' to quit).\n")


def format_table(columns, rows, max_rows=10):
    """Simple terminal table formatter without extra dependencies."""
    if not rows:
        return "  (0 rows)"

    col_widths = [len(c) for c in columns]
    for row in rows[:max_rows]:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header = " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(columns))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(columns)))
    lines = [header, separator]

    for row in rows[:max_rows]:
        line = " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        lines.append(line)

    if len(rows) > max_rows:
        lines.append(f"  ... and {len(rows) - max_rows} more rows.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query Traffic Database using Natural Language")
    parser.add_argument("--query", "-q", type=str, help="Single query to run non-interactively")
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "mock"])
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    parser.add_argument("--db", default=os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite"))
    parser.add_argument("--verbose", "-v", action="store_true", help="Show full generated SQL and execution stats")

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database file not found at: {args.db}")
        sys.exit(1)

    llm_client = EdgeLLMClient(
        provider=args.provider,
        model_name=args.model,
        endpoint=args.endpoint,
        context_size=2048,
        temperature=0.0
    )

    # Health check
    is_alive, msg = llm_client.check_health()
    if not is_alive and args.provider != "mock":
        print(f"[WARNING] LLM server warning: {msg}")
        print("Tip: If Ollama is not running, start it with 'ollama serve' or run with '--provider mock' for offline testing.\n")

    pipeline = TrafficQueryPipeline(db_path=args.db, llm_client=llm_client)

    if args.query:
        # Non-interactive single query
        res = pipeline.query(args.query)
        print(f"\nQuestion: {res['question']}")
        print(f"Generated SQL: {res['sql']}")
        print(f"Answer: {res['answer']}")
        if args.verbose and res.get("execution_result", {}).get("rows"):
            exec_res = res["execution_result"]
            print("\nRaw Results:")
            print(format_table(exec_res["columns"], exec_res["rows"]))
        print(f"Latency: {res['total_latency_seconds']} s\n")
        return

    # Interactive REPL
    print_banner(args.model, args.provider, args.db)
    while True:
        try:
            user_input = input("traffic-ai> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting Traffic Query Engine.")
                break

            res = pipeline.query(user_input)
            print(f"\n[SQL Generated]: {res['sql']}")
            if res.get("was_corrected"):
                print("  (Query was automatically corrected via self-correction loop)")

            print(f"\n[Answer]: {res['answer']}")

            if args.verbose or res.get("execution_result", {}).get("row_count", 0) > 1:
                exec_res = res.get("execution_result", {})
                if exec_res and exec_res.get("rows"):
                    print("\n[Result Table]:")
                    print(format_table(exec_res["columns"], exec_res["rows"]))

            print(f"\n[Latency]: LLM: {res['llm_latency_seconds']}s | Total: {res['total_latency_seconds']}s\n" + "-" * 50)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()

