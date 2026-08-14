#!/usr/bin/env python3
"""Wheeled-SRL 论文补充图 — 按用户推荐清单 (nature-draw.md v2.0 规范).

复用 make_paper_figures_v2.py 的样式工具 (IBM 配色 / EMA 0.02 / 无衬线 / PDF+PNG600).

推荐清单落地:
  Fig.1 fig_jump_flat_training_overview_2x2_v2 — 训练全景: 奖励/存活(回合)/动作std/loss
  Fig.2 fig_jump_flat_final_perf_2x1_v2         — 最终性能: 平均奖励 + 跳跃高度 (柱状)
  Fig.3 fig_jump_flat_traj_vel_2x1_v2           — 跳跃轨迹: 高度 + 前向速度 (2×1)
  Fig.4 fig_jump_flat_joints_2x3_v2             — 关节角: 髋pitch/膝/髋roll 左右腿 (2×3)
  Fig.5 fig_jump_flat_reward_split_2x2_v2       — 奖励分项拆解 (2×2 分组柱状)

数据:
- TB 日志 (logs/rsl_rl_ppo/XqRobotWLJump*) — Fig.1/2/5
- 扩展轨迹 npz (latex/Wheeled-SRL-Jumping/data/jump_traj_*.npz, 含 linvel/base_euler/hip_roll)
  — Fig.3/4

用法:
  uv run python tools/make_paper_supp_figs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.make_paper_figures_v2 import (  # noqa: E402
    ALGO_COLORS,
    ALGOS,
    DATA,
    FIG,
    TEXTW,
    _plot_curves,
    _style_ax,
    ema,
    load_traj,
    panel_label,
    run_dirs,
    save,
)

# ---- 补充图常量 ----
TRAJ_STEMS = ["jump_traj_srl", "jump_traj_srlvmc", "jump_traj_ppo", "jump_traj_vmc"]
TRAJ_ALGO = {"jump_traj_srl": "SRL", "jump_traj_srlvmc": "SRL+VMC",
             "jump_traj_ppo": "PPO", "jump_traj_vmc": "PPO+VMC"}

PHASES = {-1: "idle", 0: "crouch", 1: "thrust", 2: "flight", 3: "landing", 4: "recover"}
PHASE_COLORS = {-1: "#dddddd", 0: "#cfe3f4", 1: "#fde8cd", 2: "#d5efdf",
                3: "#f5dad1", 4: "#e8e8e8"}

# 4 算法共同的奖励分项 (PPO 系用 launch_rise, 故 vertical_thrust/height_progress 排除)
REWARD_TERMS = {
    "alive": "alive",
    "crouch_depth": "crouch depth",
    "jump_height": "jump height",
    "landing_soft": "landing soft",
    "orientation": "orientation",
    "lean_forward": "lean forward",
    "leg_mirror": "leg mirror",
}


def load_tb_fast(logdir: str, tags: list[str]) -> dict[str, dict]:
    """Load TensorBoard scalars once per run."""
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


def shade_phases(ax, t, phase) -> None:
    if phase.max() < 0:
        return
    uniq = [s for s in sorted(set(phase.tolist())) if s >= 0]
    for s in uniq:
        mask = phase == s
        idx = np.flatnonzero(mask)
        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in splits:
            if len(run) < 2:
                continue
            ax.axvspan(t[run[0]], t[run[-1]], color=PHASE_COLORS.get(int(s), "#eeeeee"),
                       alpha=0.5, lw=0, zorder=0)
    for s in uniq:
        mask = phase == s
        if mask.sum() < 2:
            continue
        tm = float(np.median(t[mask]))
        ax.text(tm, 1.02, PHASES[int(s)], transform=ax.get_xaxis_transform(),
                ha="center", fontsize=6, color="#444444", style="italic")


def _legend_below(fig, axes) -> None:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=4, frameon=False, fontsize=7)


# ================= Fig.1 训练全景 2×2 =================

def fig_training_overview(dirs) -> None:
    panels = [
        ("Train/mean_reward", "Mean Reward", True),
        ("Train/mean_episode_length", "Episode Length", True),
        ("Policy/mean_std", "Action Std", False),
        ("Loss/value", "Value Loss", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, TEXTW * 0.62), layout="constrained")
    for ax, (tag, title, smooth_it) in zip(axes.flatten(), panels):
        _plot_curves(ax, dirs, tag, smooth_it=smooth_it)
        panel_label(ax, f"({chr(97 + panels.index((tag, title, smooth_it)))})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training iteration (×10$^3$)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        _style_ax(ax)
    _legend_below(fig, axes)
    fig.tight_layout(pad=0.6, rect=(0, 0.06, 1, 1))
    save(fig, "fig_jump_flat_training_overview_2x2_v2")


# ================= Fig.2 最终性能 2×1 (奖励 + 跳高) =================

def fig_final_perf(dirs) -> None:
    # 最终平均奖励 = 末 500 iter 均值 (EMA 后更稳)
    final_reward = {}
    for algo in ALGOS:
        data = load_tb_fast(dirs[algo], ["Train/mean_reward"]).get("Train/mean_reward")
        if data is None:
            final_reward[algo] = 0.0
            continue
        v = ema(data["values"], 0.02)[-500:]
        final_reward[algo] = float(v.mean())
    # 跳高来自 four_algo_comparison.json
    import json
    cmp = json.load(open(DATA / "four_algo_comparison.json"))["results"]
    keys = {"SRL": "XqRobotWLJumpSRLFlat", "SRL+VMC": "XqRobotWLJumpSRLVMC",
            "PPO": "XqRobotWLJumpFlat", "PPO+VMC": "XqRobotWLJumpVMC"}
    jump_h = [cmp[k]["jump_height"] for k in keys.values()]

    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, TEXTW * 0.34), layout="constrained")
    x = np.arange(len(ALGOS))
    for ax, vals, ylabel in zip(axes, (jump_h, [final_reward[a] for a in ALGOS]),
                                ("Jump Height (m)", "Final Mean Reward")):
        ax.bar(x, vals, width=0.6, color=[ALGO_COLORS[a] for a in ALGOS],
               edgecolor="white", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (max(vals) * 0.02), f"{v:.3f}", ha="center",
                    fontsize=7, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels(ALGOS, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        _style_ax(ax)
    for ax, lab in zip(axes, "ab"):
        panel_label(ax, f"({lab})")
    save(fig, "fig_jump_flat_final_perf_2x1_v2")


# ================= Fig.3 跳跃轨迹: 高度 + 前向速度 2×1 =================

def fig_traj_vel() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, TEXTW * 0.4), sharex=True, layout="constrained")
    for ax, (key, ylab) in zip(axes, [("base_z", "Base Height (m)"),
                                      ("linvel", "Forward Velocity (m/s)")]):
        for st in TRAJ_STEMS:
            d = load_traj(st)
            t = d["t"]
            val = d["linvel"][:, 0] if key == "linvel" else d["base_z"]
            ax.plot(t, val, color=ALGO_COLORS[TRAJ_ALGO[st]], lw=1.5, label=TRAJ_ALGO[st])
            if st == "jump_traj_ppo" and bool(np.asarray(d.get("terminated", [False]))[0]):
                ax.plot(t[-1], val[-1], marker="x", ms=7, mew=1.5,
                        color=ALGO_COLORS["PPO"], zorder=4)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        _style_ax(ax)
    for ax, lab in zip(axes, "ab"):
        panel_label(ax, f"({lab})")
    _legend_below(fig, axes)
    fig.tight_layout(pad=0.6, rect=(0, 0.10, 1, 1))
    save(fig, "fig_jump_flat_traj_vel_2x1_v2")


# ================= Fig.4 关节角 2×3 (髋pitch/膝/髋roll) =================

def fig_joints_2x3() -> None:
    # 2 行 × 3 列: 行=SRL(上) 与 PPO(下), 列=髋pitch/膝/髋roll
    chosen = ["jump_traj_srl", "jump_traj_ppo"]
    rows = [TRAJ_ALGO[s] for s in chosen]
    cols = [("hip_pitch", "Hip Pitch (deg)"),
            ("knee", "Knee (deg)"),
            ("hip_roll", "Hip Roll (deg)")]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, TEXTW * 0.62), sharex=True,
                             layout="constrained")
    for r, stem in enumerate(chosen):
        d = load_traj(stem)
        t = d["t"]
        for c, (key, _) in enumerate(cols):
            ax = axes[r, c]
            col = ALGO_COLORS[TRAJ_ALGO[stem]]
            # 左右腿: L 实线, R 虚线
            ax.plot(t, np.degrees(d[key][:, 0]), color=col, lw=1.4, ls="-", label="L")
            ax.plot(t, np.degrees(d[key][:, 1]), color=col, lw=1.4, ls="--", label="R")
            ax.set_ylim(-120, 120)
            _style_ax(ax)
    for r, name in enumerate(rows):
        axes[r, 0].set_ylabel(name, fontsize=9)
        axes[r, 0].set_ylabel(f"{name}  |  Joint Angle (deg)", fontsize=9)
    for c, (_, ylab) in enumerate(cols):
        axes[0, c].set_title(ylab.split(" (")[0], fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=9)
    # 子图编号 (a)-(f) 行优先
    for i, ax in enumerate(axes.flatten()):
        panel_label(ax, f"({chr(97 + i)})")
    axes[0, 0].legend(fontsize=7, ncol=2, loc="upper right", frameon=False)
    save(fig, "fig_jump_flat_joints_2x3_v2")


# ================= Fig.5 奖励分项拆解 2×2 =================

def fig_reward_split(dirs) -> None:
    # 每算法末 500 iter 各 reward term 均值; 4 面板 (每算法一图, 按 term 排序)
    terms = list(REWARD_TERMS)
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, TEXTW * 0.62), layout="constrained")
    for ax, algo in zip(axes.flatten(), ALGOS):
        data = load_tb_fast(dirs[algo], [f"reward/{t}" for t in terms])
        vals = []
        for t in terms:
            d = data.get(f"reward/{t}")
            vals.append(float(ema(d["values"], 0.02)[-500:].mean()) if d is not None else 0.0)
        order = np.argsort(vals)[::-1]
        x = np.arange(len(terms))
        ax.bar(x[order], np.array(vals)[order], width=0.6, color=ALGO_COLORS[algo],
               edgecolor="white", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([REWARD_TERMS[terms[i]] for i in order], fontsize=6.5,
                           rotation=30, ha="right")
        ax.set_ylabel("Mean Reward", fontsize=9)
        ax.set_title(algo, fontsize=9)
        _style_ax(ax)
        for xi, v in zip(x[order], np.array(vals)[order]):
            if v > 0.01:
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5,
                        color="#222222")
    for i, ax in enumerate(axes.flatten()):
        panel_label(ax, f"({chr(97 + i)})")
    save(fig, "fig_jump_flat_reward_split_2x2_v2")


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    dirs = run_dirs()
    print(f"训练目录: {len(dirs)}/4, 输出 → {FIG}")
    print("[1/5] Fig.1 训练全景 2×2 (奖励/存活/std/loss) ...")
    fig_training_overview(dirs)
    print("[2/5] Fig.2 最终性能 2×1 (跳高+奖励) ...")
    fig_final_perf(dirs)
    print("[3/5] Fig.3 轨迹高度+前向速度 2×1 ...")
    fig_traj_vel()
    print("[4/5] Fig.4 关节角 2×3 (髋pitch/膝/髋roll) ...")
    fig_joints_2x3()
    print("[5/5] Fig.5 奖励分项拆解 2×2 ...")
    fig_reward_split(dirs)
    print("补充图完成.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
