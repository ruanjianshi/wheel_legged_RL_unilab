#!/usr/bin/env python3
"""图3: 消融实验对比柱状图.

横轴: full / no_fsm / no_wheel_match / no_flight_mod / no_vel_track
纵轴: 跳跃成功率 / 平均跳距 / 着陆轮滑
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np


def plot_ablation(
    results_base: str = "results",
    out_path: str = "results/ablation_comparison.png",
):
    """绘制消融实验对比."""
    models = {
        "Wheeled-SRL": "srl_full",
        "No FSM": "srl_no_fsm",
        "No Wheel Match": "srl_no_wheel_match",
        "No Flight Mod": "srl_no_flight_mod",
        "No Vel Track": "srl_no_vel_track",
    }

    # 加载各模型汇总
    metrics_data = {}
    for label, dir_name in models.items():
        try:
            with open(f"{results_base}/{dir_name}/summary.json") as f:
                metrics_data[label] = json.load(f)
        except FileNotFoundError:
            metrics_data[label] = {"success_rate": 0, "avg_jump_distance": 0, "avg_wheel_slip": 1.0}

    labels = list(models.keys())
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # (a) Success Rate
    sr = [metrics_data.get(l, {}).get("success_rate", 0) for l in labels]
    bars = axes[0].bar(x, sr, width, color="#4CAF50")
    axes[0].set_title("(a) Success Rate")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    for bar, v in zip(bars, sr):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.2f}",
            ha="center",
            fontsize=8,
        )

    # (b) Avg Jump Distance
    jd = [metrics_data.get(l, {}).get("avg_jump_distance", 0) for l in labels]
    bars = axes[1].bar(x, jd, width, color="#2196F3")
    axes[1].set_title("(b) Avg Jump Distance")
    axes[1].set_ylabel("Distance (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    for bar, v in zip(bars, jd):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=8,
        )

    # (c) Wheel Slip
    ws = [min(metrics_data.get(l, {}).get("avg_wheel_slip", 1.0), 1.0) for l in labels]
    bars = axes[2].bar(x, ws, width, color="#F44336")
    axes[2].set_title("(c) Avg Wheel Slip at Landing")
    axes[2].set_ylabel("Slip (m/s)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=15)
    for bar, v in zip(bars, ws):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            fontsize=8,
        )

    fig.suptitle("Ablation Study", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys

    base = sys.argv[1] if len(sys.argv) > 1 else "results"
    plot_ablation(base)
