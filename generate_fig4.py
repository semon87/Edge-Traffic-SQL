#!/usr/bin/env python3
"""
generate_fig4.py
Generates Figure 4 (Grouped Bar Chart of Execution Accuracy by Difficulty Tier)
with Wilson 95% Confidence Intervals for IEEEtran (single-column width).
Outputs: fig_accuracy_tiers.pdf and fig_accuracy_tiers.png
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

categories = ["Simple\n(N=20)", "Intermediate\n(N=20)", "Advanced\n(N=20)", "Overall\n(N=60)"]
strict_ex = [100.0, 85.0, 60.0, 81.67]
soft_ex   = [100.0, 95.0, 90.0, 95.00]

# Wilson 95% Confidence Intervals for Overall (N=60)
# Strict EX (49/60): [70.08%, 89.44%] -> lower diff: 11.59, upper diff: 7.77
# Soft EX (57/60):   [86.30%, 98.29%] -> lower diff: 8.70,  upper diff: 3.29
strict_err = np.zeros((2, 4))
strict_err[0, 3] = 11.59
strict_err[1, 3] = 7.77

soft_err = np.zeros((2, 4))
soft_err[0, 3] = 8.70
soft_err[1, 3] = 3.29

x = np.arange(len(categories))
width = 0.35  # width of each bar

# Standard single column width in IEEEtran is ~3.5 inches (8.8 cm)
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

# Colors matching IEEE aesthetic
c_strict = "#2b5c8f"  # Deep Steel Blue
c_soft   = "#388e3c"  # Muted Forest Green

rects1 = ax.bar(x - width/2, strict_ex, width, label="Strict EX (%)", 
                color=c_strict, edgecolor="black", linewidth=0.8,
                yerr=strict_err, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"})

rects2 = ax.bar(x + width/2, soft_ex, width, label="Semantic Intent EX (%)", 
                color=c_soft, edgecolor="black", linewidth=0.8,
                yerr=soft_err, capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "black"})

# Labels and styling
ax.set_ylabel("Execution Accuracy (%)", fontweight="bold")
ax.set_ylim(0, 115)
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Place legend cleanly outside above plot to avoid covering any bars or error caps
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
          frameon=True, framealpha=0.95, edgecolor="#cccccc",
          columnspacing=0.8, handletextpad=0.3)

ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.set_axisbelow(True)

# Value labels positioned cleanly above bars or above upper error bar caps
for idx in range(len(categories)):
    # Strict EX label
    h1 = strict_ex[idx]
    top1 = h1 + strict_err[1, idx]
    t1 = f"{h1:.1f}" if h1 != 100.0 else "100"
    ax.annotate(t1, xy=(x[idx] - width/2, top1), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    
    # Soft EX label
    h2 = soft_ex[idx]
    top2 = h2 + soft_err[1, idx]
    t2 = f"{h2:.1f}" if h2 != 100.0 else "100"
    ax.annotate(t2, xy=(x[idx] + width/2, top2), xytext=(0, 2),
                textcoords="offset points", ha="center", va="bottom", fontsize=6.8, fontweight="bold")

plt.tight_layout()
plt.savefig("fig_accuracy_tiers.pdf", bbox_inches="tight")
plt.savefig("fig_accuracy_tiers.png", bbox_inches="tight")
print("Saved fig_accuracy_tiers.pdf and fig_accuracy_tiers.png successfully!")
