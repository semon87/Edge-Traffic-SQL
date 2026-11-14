#!/usr/bin/env python3
"""
generate_fig5.py
Generates Figure 5 (Read latency under concurrent ingestion, Rollback vs WAL).
Outputs: fig_wal_latency.pdf and fig_wal_latency.png
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication style font matching IEEEtran
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9.5
})

metrics = ["Mean", "P50\n(Median)", "P95", "P99"]
rollback_vals = [2.58, 0.81, 16.50, 31.42]
wal_vals      = [0.78, 0.75,  1.57,  2.59]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

c_rollback = "#d9534f"  # Crimson / Coral red
c_wal      = "#2e7d32"  # Deep Forest Green

rects1 = ax.bar(x - width/2, rollback_vals, width, label="Rollback Journal",
                color=c_rollback, edgecolor="black", linewidth=0.8)
rects2 = ax.bar(x + width/2, wal_vals, width, label="Write-Ahead Logging (WAL)",
                color=c_wal, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Read Latency (ms, Log Scale)", fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(0.2, 70)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
ax.set_xticks(x)
ax.set_xticklabels(metrics)

# Place legend cleanly outside above plot to prevent obscuring any bar or value labels
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=True, framealpha=0.95, edgecolor="#cccccc",
          columnspacing=0.8, handletextpad=0.3)

ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

def autolabel_log(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

autolabel_log(rects1)
autolabel_log(rects2)

plt.tight_layout()
plt.savefig("fig_wal_latency.pdf", bbox_inches="tight")
plt.savefig("fig_wal_latency.png", bbox_inches="tight")
print("Saved fig_wal_latency.pdf and fig_wal_latency.png successfully!")
