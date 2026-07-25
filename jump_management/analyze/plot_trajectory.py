#!/usr/bin/env python3
"""图2: 不同距离跳跃轨迹追踪对比.

横轴: 时间步, 纵轴: 机身高度
三子图: fix_01m / fix_02m / fix_03m
每子图: PPO-only vs Wheeled-SRL 各一条
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(
    ppo_results_dir: str,
    srl_results_dir: str,
    out_path: str = "results/trajectory_comparison.png",
):
    """绘制不同距离跳跃轨迹对比."""
    scenarios = ["fix_01m", "fix_02m", "fix_03m"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"PPO-only": "#2196F3", "Wheeled-SRL": "#F44336"}

    for i, scenario in enumerate(scenarios):
        ax = axes[i]

        for results_dir, label in [
            (ppo_results_dir, "PPO-only"),
            (srl_results_dir, "Wheeled-SRL"),
        ]:
            try:
                with open(f"{results_dir}/{scenario}.json") as f:
                    data = json.load(f)
            except FileNotFoundError:
                continue

            # 提取跳距和高度
            distances = []
            heights = []
            for ep in data:
                for j in ep.get("jump_details", []):
                    distances.append(j["distance"])
                    heights.append(j["max_height"])

            if distances:
                ax.scatter(distances, heights, label=label, color=colors[label], alpha=0.6, s=30)

        ax.set_xlabel("Jump Distance (m)")
        ax.set_ylabel("Max Height (m)")
        ax.set_title(scenario)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Jump Trajectory Scatter: Distance vs Height", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys

    ppo_dir = sys.argv[1] if len(sys.argv) > 1 else "results/ppo_only"
    srl_dir = sys.argv[2] if len(sys.argv) > 2 else "results/srl_full"
    plot_trajectory(ppo_dir, srl_dir)
