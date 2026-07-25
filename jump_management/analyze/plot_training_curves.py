#!/usr/bin/env python3
"""图1: PPO-only vs Wheeled-SRL 训练曲线对比.

从 TensorBoard 日志读取, 生成四子图:
  (a) mean_reward  (b) episode_length
  (c) jump_height reward  (d) action_std
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_metrics(logdir: str, tags: list[str]) -> dict[str, np.ndarray]:
    """加载 TensorBoard scalar 数据."""
    ea = EventAccumulator(logdir)
    ea.Reload()

    result = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        if scalars:
            vals = np.array([s.value for s in scalars])
            steps = np.array([s.step for s in scalars])
            result[tag] = {"values": vals, "steps": steps}
    return result


def find_latest_run(base_dir: str) -> str | None:
    import glob

    runs = glob.glob(f"{base_dir}/2*/")
    if not runs:
        return None
    return sorted(runs)[-1]


def plot_training_curves(
    ppo_dir: str,
    srl_dir: str,
    out_path: str = "results/training_curves.png",
):
    """绘制训练曲线对比图."""
    tags = [
        "reward/mean",
        "episode_length/mean",
        "reward/jump_height",
        "reward/jump_height",  # placeholder for action_std
        "std/mean",
    ]

    ppo_run = find_latest_run(ppo_dir)
    srl_run = find_latest_run(srl_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    labels = ["(a) Mean Reward", "(b) Episode Length", "(c) Jump Height Reward", "(d) Action Std"]
    colors = {"PPO-only": "#2196F3", "Wheeled-SRL": "#F44336"}

    for run, run_name, run_label in [
        (ppo_run, "ppo", "PPO-only"),
        (srl_run, "srl", "Wheeled-SRL"),
    ]:
        if not run:
            continue
        metrics = load_tb_metrics(
            run,
            [
                "reward/mean",
                "episode_length/mean",
                "reward/jump_height",
                "std/mean",
            ],
        )

        for i, (tag, label) in enumerate(
            zip(
                ["reward/mean", "episode_length/mean", "reward/jump_height", "std/mean"],
                labels,
            )
        ):
            if tag in metrics:
                axes[i].plot(
                    metrics[tag]["steps"],
                    metrics[tag]["values"],
                    label=run_label,
                    color=colors[run_label],
                    linewidth=1.0,
                )

    for i, ax in enumerate(axes):
        ax.set_xlabel("Steps")
        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import sys

    ppo_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl_ppo/XqRobotWLJumpFlat"
    srl_dir = sys.argv[2] if len(sys.argv) > 2 else "logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat"
    plot_training_curves(ppo_dir, srl_dir)
