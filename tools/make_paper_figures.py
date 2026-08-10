#!/usr/bin/env python3
"""生成论文图: 训练对比 (2x2) + 图4.1 训练指标 (2x3) + 验证指标 (两栏柱状).

按 dataviz/nature 规范统一风格:
- Okabe-Ito 四色经 scripts/validate_palette.js 验证 (CVD 全过; #E69F00 对比度
  WARN 由黑色图例文字缓解);
- 每面板单 y 轴 (动作标准差与 FPS 拆成独立面板, 不画双轴);
- constrained layout (避免 tight_layout/gridspec 警告), 衰退网格, 细脊线。

Figure 1 (2x2, paper_fig_training): 训练对比 (EMA 平滑, --smooth 默认 0.8)
  mean_reward / mean_episode_length / jump_height reward / landing_soft reward
Figure 2 (2x3, paper_fig_training_metrics): 图 4.1 训练指标, 四算法同色并排
  (a) Mean Reward (b) Episode Length (c) Jump Height Reward (d) Landing Soft
  Reward (e) Action Std (f) Training FPS; reward 类曲线 EMA 0.8, std/FPS 原始。
Figure 3 (单张图, paper_fig_validation): Air Fraction + Survival Rate 柱状。

高度变化曲线由 scripts/plot_jump_trajectory.py 按参考风格 (2x2 + FSM 相位色带)
单独出 paper_fig_trajectory。四种算法统一 Okabe-Ito 色盲安全色 (论文标准)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401  # scientific paper styles

# 科研样式: science + ieee (黑白可读, 单栏宽度), no-latex (系统无 LaTeX)
plt.style.use(["science", "ieee", "no-latex"])
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Okabe-Ito 色盲安全色
COLORS = {
    "SRL": "#0072B2",  # blue
    "PPO": "#E69F00",  # orange
    "PPO+VMC": "#009E73",  # green
    "VMC+SRL": "#D55E00",  # vermillion
}

ALGOS = ["SRL", "PPO", "PPO+VMC", "VMC+SRL"]


def ema_smooth(values: np.ndarray, s: float = 0.8) -> np.ndarray:
    """TensorBoard-style exponential smoothing: y_i = s*y_{i-1} + (1-s)*x_i.

    s 越大越平滑 (0.8 ≈ 只看趋势); 对后期发散/毛刺多的曲线 (如纯 PPO) 效果显著。
    """
    out = np.empty_like(values)
    if len(values) == 0:
        return out
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = s * out[i - 1] + (1.0 - s) * values[i]
    return out


def _style_ax(ax, grid_alpha: float = 0.2) -> None:
    """Nature 风格: 衰退网格 + 细脊线 (配色经 validate_palette.js 验证)."""
    ax.grid(alpha=grid_alpha, lw=0.4, color="#bbbbbb")
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def _plot_curves(ax, dirs, tag, smooth, smooth_it=True) -> None:
    """四算法同色并排画一条曲线; 返回 False 表示无数据."""
    any_data = False
    for algo in ALGOS:
        if algo not in dirs:
            continue
        data = load_tb(dirs[algo], [tag]).get(tag)
        if data is None or len(data["values"]) == 0:
            continue
        y = ema_smooth(data["values"], smooth) if smooth_it else data["values"]
        ax.plot(data["steps"], y, color=COLORS[algo], lw=1.6, label=algo)
        any_data = True
    return any_data


def load_tb(logdir: str, tags: list[str]) -> dict[str, dict]:
    """Load TensorBoard scalars: {tag: {"steps": np, "values": np}}."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    out = {}
    for tag in tags:
        try:
            scalars = ea.Scalars(tag)
        except KeyError:
            continue
        if scalars:
            out[tag] = {
                "steps": np.array([s.step for s in scalars]),
                "values": np.array([s.value for s in scalars]),
            }
    return out


def run_dirs() -> dict[str, dict[str, str]]:
    """Auto-discover run dirs per algorithm + training tags."""
    base = ROOT / "logs" / "rsl_rl_ppo"
    cfg = {
        "SRL": ("XqRobotWLJumpSRLFlat", "2026-08-06_01-16-20_mujoco"),
        "PPO": (
            "XqRobotWLJumpFlat",
            "2026-08-09_01-21-11_mujoco",
        ),  # v12 launch_rise (best ckpt model_4000)
        "PPO+VMC": ("XqRobotWLJumpVMC", "2026-08-09_01-21-12_mujoco"),  # v12 launch_rise
        "VMC+SRL": ("XqRobotWLJumpSRLVMC", "2026-08-08_01-05-51_mujoco"),  # v4 ±50+long FSM
    }
    found = {}
    for algo, (task, run) in cfg.items():
        task_dir = base / task
        if not task_dir.exists():
            print(f"WARN: no runs for {algo} ({task})")
            continue
        if run is None:
            runs = sorted([d for d in task_dir.iterdir() if d.is_dir()])
            run_dir = runs[-1] if runs else None
        else:
            run_dir = task_dir / run
        if run_dir and run_dir.exists():
            found[algo] = str(run_dir)
    return found


def fig_training(dirs: dict[str, str], out_path: Path, smooth: float = 0.8) -> None:
    """Figure 1: 2x2 训练对比曲线 (EMA 平滑, nature 风格)."""
    tags = [
        ("Train/mean_reward", "Mean Reward"),
        ("Train/mean_episode_length", "Episode Length"),
        ("reward/jump_height", "Jump Height Reward"),
        ("reward/landing_soft", "Landing Soft Reward"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
    for ax, (tag, title) in zip(axes.flatten(), tags):
        _plot_curves(ax, dirs, tag, smooth)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Iteration", fontsize=7.5)
        ax.set_ylabel("Value", fontsize=7.5)
        ax.tick_params(labelsize=7)
        _style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        handletextpad=0.4,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"训练对比图 -> {out_path}")


def fig_training_metrics(dirs: dict[str, str], out_path: Path, smooth: float = 0.8) -> None:
    """Figure 2 (论文图 4.1): 2x3 训练指标, 四算法同色并排.

    (a) Mean Reward (b) Episode Length (c) Jump Height Reward
    (d) Landing Soft Reward (e) Action Std (f) Training FPS.
    reward 类曲线 EMA 平滑 (噪声大), 动作标准差与 FPS 用原始值 (各自单轴, 不画双轴).
    """
    panels = [
        ("Train/mean_reward", "Mean Reward", True),
        ("Train/mean_episode_length", "Episode Length", True),
        ("reward/jump_height", "Jump Height Reward", True),
        ("reward/landing_soft", "Landing Soft Reward", True),
        ("Policy/mean_std", "Action Std", False),
        ("Perf/total_fps", "Training FPS", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), layout="constrained")
    for ax, (tag, title, smooth_it) in zip(axes.flatten(), panels):
        _plot_curves(ax, dirs, tag, smooth, smooth_it=smooth_it)
        ax.set_title(f"({chr(97 + panels.index((tag, title, smooth_it)))}) {title}", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=7.5)
        ax.set_ylabel("Value", fontsize=7.5)
        ax.tick_params(labelsize=7)
        _style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        handletextpad=0.4,
    )
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"图4.1 训练指标图 -> {out_path}")


def fig_validation(out_path: Path) -> None:
    """Figure 2: 单张对比图 — 腾空 + 存活, 四种算法同色并排 (高度曲线见 paper_fig_trajectory)."""
    compare_json = ROOT / "jump_management" / "results" / "four_algo_comparison.json"
    if compare_json.exists():
        data = json.load(open(compare_json))["results"]
        algos = [
            "XqRobotWLJumpSRLFlat",
            "XqRobotWLJumpFlat",
            "XqRobotWLJumpVMC",
            "XqRobotWLJumpSRLVMC",
        ]
        labels = ["SRL", "PPO", "PPO+VMC", "VMC+SRL"]
        airs = [data[a]["air_frac"] for a in algos if a in data]
        survs = [data[a]["survival"] for a in algos if a in data]
    else:
        print("WARN: four_algo_comparison.json not found; skipping validation figure")
        return

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.6), layout="constrained")
    for ax, vals, ylabel in zip(axes, (airs, survs), ("Air Fraction", "Survival Rate")):
        ax.bar(x, vals, width=0.6, color=[COLORS[l] for l in labels], edgecolor="white", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7.5, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(0, max(max(vals) * 1.2, 1.0))
        ax.grid(axis="y", alpha=0.2, lw=0.4, color="#bbbbbb", zorder=0)
        for s in ax.spines.values():
            s.set_linewidth(0.6)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"验证指标图 (腾空/存活) -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "jump_management" / "results"))
    parser.add_argument(
        "--smooth", type=float, default=0.8, help="EMA 平滑系数 (TensorBoard 风格, 默认 0.8)"
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = run_dirs()
    print("发现训练目录:", {k: Path(v).name for k, v in dirs.items()})
    print(f"EMA 平滑: {args.smooth}")
    fig_training(dirs, out_dir / "paper_fig_training", args.smooth)
    fig_training_metrics(dirs, out_dir / "paper_fig_training_metrics", args.smooth)
    fig_validation(out_dir / "paper_fig_validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
