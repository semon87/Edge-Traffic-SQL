"""
web_ui.py
Lightweight Zero-Dependency Web Dashboard for Edge Traffic Query Engine.

Runs using Python standard library (http.server, json, urllib).
Consumes virtually zero RAM (< 35 MB) on NVIDIA Jetson Orin Nano.
Access from any browser on the local network at: http://<jetson-ip>:8501
"""

import http.server
import socketserver
import json
import urllib.parse
import os
import sys

# Ensure parent directory is in sys.path so track2_query_engine is always found
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from track2_query_engine.llm_client import EdgeLLMClient
from track2_query_engine.pipeline import TrafficQueryPipeline

PORT = 8501
DB_PATH = os.path.join("outputs", "jetson_tuned", "traffic_analytics.sqlite")

pipeline_instance = None


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Edge Traffic AI - Natural Language Query Interface</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #34d399;
      --error: #f87171;
      --code-bg: #090d16;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 24px; display: flex; justify-content: center; }
    .container { max-width: 1000px; width: 100%; display: flex; flex-direction: column; gap: 20px; }
    header { border-bottom: 1px solid var(--border); padding-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    h1 { font-size: 24px; font-weight: 700; color: var(--text); display: flex; align-items: center; gap: 10px; }
    .badge { background: #0369a1; color: #e0f2fe; font-size: 12px; padding: 4px 8px; border-radius: 6px; font-weight: 600; }
    .device-tag { color: var(--text-muted); font-size: 13px; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .query-box { display: flex; gap: 12px; margin-top: 10px; }
    input[type="text"] {
      flex: 1; padding: 14px 18px; border-radius: 8px; border: 1px solid var(--border);
      background: #090d16; color: #fff; font-size: 16px; outline: none; transition: border-color 0.2s;
    }
    input[type="text"]:focus { border-color: var(--accent); }
    button {
      padding: 14px 28px; background: var(--accent); color: #0f172a; font-weight: 700;
      font-size: 15px; border: none; border-radius: 8px; cursor: pointer; transition: background 0.2s;
    }
    button:hover { background: var(--accent-hover); color: #fff; }
    .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .pill {
      background: #1e293b; border: 1px solid var(--border); color: var(--text-muted);
      font-size: 12px; padding: 6px 12px; border-radius: 16px; cursor: pointer; transition: all 0.2s;
    }
    .pill:hover { border-color: var(--accent); color: var(--accent); background: #0c4a6e; }
    .results-area { display: flex; flex-direction: column; gap: 16px; }
    .answer-card { background: #0f233a; border-left: 4px solid var(--accent); padding: 16px; border-radius: 6px; font-size: 16px; line-height: 1.5; }
    .code-block { background: var(--code-bg); padding: 16px; border-radius: 8px; font-family: monospace; font-size: 14px; overflow-x: auto; color: #38bdf8; border: 1px solid var(--border); }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }
    th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--border); }
    th { background: #162032; color: var(--accent); font-weight: 600; text-transform: uppercase; font-size: 12px; }
    tr:hover { background: #24344d; }
    .stats-bar { display: flex; gap: 20px; font-size: 13px; color: var(--text-muted); margin-top: 10px; }
    .spinner { display: none; text-align: center; padding: 20px; color: var(--accent); font-size: 14px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Edge Traffic AI <span class="badge">Track 2: Text-to-SQL</span></h1>
        <div class="device-tag">Target: NVIDIA Jetson Orin Nano (8GB Unified Memory)</div>
      </div>
      <div><span class="badge" style="background:#065f46; color:#d1fae5;">SQLite WAL Active</span></div>
    </header>

    <div class="card">
      <label style="font-size:14px; font-weight:600; color:var(--text-muted);">Ask a Natural Language Traffic Query:</label>
      <div class="query-box">
        <input type="text" id="queryInput" placeholder="e.g. today between 3-4 pm which vehicle with red color go left" />
        <button onclick="submitQuery()">Execute</button>
      </div>

      <div class="suggestions">
        <span class="pill" onclick="setQuery(this.innerText)">today between 3-4 pm which vehicle with red color go left</span>
        <span class="pill" onclick="setQuery(this.innerText)">How many speeding vehicles were detected grouped by type?</span>
        <span class="pill" onclick="setQuery(this.innerText)">Which hour today had the highest total vehicle traffic volume?</span>
        <span class="pill" onclick="setQuery(this.innerText)">What is the average speed of trucks compared to cars?</span>
        <span class="pill" onclick="setQuery(this.innerText)">Find the top 3 fastest vehicles recorded across all sequences</span>
      </div>
    </div>

    <div class="spinner" id="spinner">Generating SQL & Executing on Edge Database...</div>

    <div class="results-area" id="resultsArea" style="display:none;">
      <div class="card">
        <h3 style="font-size:16px; margin-bottom:12px; color:var(--text-muted);">Synthesized Answer</h3>
        <div class="answer-card" id="answerText"></div>

        <h3 style="font-size:16px; margin-top:20px; margin-bottom:8px; color:var(--text-muted);">Generated SQLite Query</h3>
        <div class="code-block" id="sqlBlock"></div>

        <div class="stats-bar" id="statsBar"></div>

        <h3 style="font-size:16px; margin-top:20px; margin-bottom:8px; color:var(--text-muted);">Query Result Records</h3>
        <div style="overflow-x:auto;">
          <table id="resultsTable">
            <thead id="tableHead"></thead>
            <tbody id="tableBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    function setQuery(text) {
      document.getElementById('queryInput').value = text;
      submitQuery();
    }

    document.getElementById('queryInput').addEventListener('keypress', function (e) {
      if (e.key === 'Enter') submitQuery();
    });

    async function submitQuery() {
      const q = document.getElementById('queryInput').value.trim();
      if (!q) return;

      document.getElementById('spinner').style.display = 'block';
      document.getElementById('resultsArea').style.display = 'none';

      try {
        const resp = await fetch('/api/query?q=' + encodeURIComponent(q));
        const data = await resp.json();

        document.getElementById('spinner').style.display = 'none';
        document.getElementById('resultsArea').style.display = 'block';

        document.getElementById('answerText').innerHTML = data.answer.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
        document.getElementById('sqlBlock').innerText = data.sql || 'No SQL generated';

        document.getElementById('statsBar').innerHTML = 
          `<span>⏱️ Total Latency: <strong>${data.total_latency_seconds}s</strong></span>` +
          `<span>🧠 LLM Latency: <strong>${data.llm_latency_seconds}s</strong></span>` +
          `<span>📊 Matching Rows: <strong>${data.execution_result ? data.execution_result.row_count : 0}</strong></span>` +
          (data.was_corrected ? `<span style="color:#fbbf24;">⚠️ Auto-corrected via feedback loop</span>` : '');

        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');
        tableHead.innerHTML = '';
        tableBody.innerHTML = '';

        if (data.execution_result && data.execution_result.columns && data.execution_result.columns.length > 0) {
          const headerRow = document.createElement('tr');
          data.execution_result.columns.forEach(col => {
            const th = document.createElement('th');
            th.innerText = col.replace('_', ' ');
            headerRow.appendChild(th);
          });
          tableHead.appendChild(headerRow);

          data.execution_result.rows.slice(0, 15).forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(val => {
              const td = document.createElement('td');
              td.innerText = (typeof val === 'number') ? val.toFixed(1) : val;
              tr.appendChild(td);
            });
            tableBody.appendChild(tr);
          });
        }
      } catch (err) {
        document.getElementById('spinner').style.display = 'none';
        alert('Query failed: ' + err);
      }
    }
  </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/api/query":
            params = urllib.parse.parse_qs(parsed.query)
            user_q = params.get("q", [""])[0]

            if not user_q:
                res = {"error": "Empty query"}
            else:
                res = pipeline_instance.query(user_q)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Concise logging
        print(f"[WebUI] {self.address_string()} - {args[0]}")


def start_server(provider="ollama", model="qwen2.5-coder:3b", endpoint="http://localhost:11434"):
    global pipeline_instance
    print(f"Initializing Edge Pipeline (Model: {model}, Provider: {provider})...")
    llm_client = EdgeLLMClient(provider=provider, model_name=model, endpoint=endpoint)
    pipeline_instance = TrafficQueryPipeline(DB_PATH, llm_client=llm_client)

    server_address = ("", PORT)
    with socketserver.TCPServer(server_address, DashboardHandler) as httpd:
        print("\n" + "=" * 65)
        print(f"  Edge Traffic AI Dashboard is LIVE!")
        print(f"  Local Access:    http://localhost:{PORT}")
        print(f"  Network Access:  http://<jetson-ip>:{PORT}")
        print("=" * 65 + "\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down Web Dashboard.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "mock"])
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--endpoint", default="http://localhost:11434")
    args = parser.parse_args()

    start_server(provider=args.provider, model=args.model, endpoint=args.endpoint)

