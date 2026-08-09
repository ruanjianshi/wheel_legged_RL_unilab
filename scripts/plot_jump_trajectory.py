#!/usr/bin/env python3
"""Paper figures for a single jump process of each algorithm (2x2 layout).

Figure A (2x2, paper_fig_trajectory): base_z height curve per algorithm,
  with SLIP-FSM phase bands; terminated episodes (pure PPO) get an end marker.
Figure B (2x2, paper_fig_jump_joints): joint angles per algorithm
  (knee L/R solid/dashed, hip pitch L/R) across the same single jump.

Both figures share a time axis so the crouch -> thrust -> flight -> landing
sequence is directly comparable across algorithms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401

plt.style.use(["science", "ieee", "no-latex"])

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "jump_management" / "results"

# Okabe-Ito colourblind-safe palette (order matches ALGOS)
COLORS = {
    "SRL": "#0072B2",
    "PPO": "#E69F00",
    "PPO+VMC": "#009E73",
    "VMC+SRL": "#D55E00",
}

# (npz stem, figure label)
ALGOS = [
    ("jump_traj_srl", "SRL"),
    ("jump_traj_ppo", "PPO"),
    ("jump_traj_vmc", "PPO+VMC"),
    ("jump_traj_srlvmc", "VMC+SRL"),
]

# SLIP-FSM phase names (jump_srl._update_fsm_state)
PHASES = {-1: "idle", 0: "crouch", 1: "thrust", 2: "flight", 3: "landing", 4: "recover"}
PHASE_COLORS = {
    -1: "#dddddd",
    0: "#cfe3f4",  # light blue
    1: "#fde8cd",  # light orange
    2: "#d5efdf",  # light green
    3: "#f5dad1",  # light vermillion
    4: "#e8e8e8",
}


def load_traj(name: str) -> dict:
    d = np.load(RESULTS / f"{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def shade_phases(ax, t, phase, xmax) -> None:
    """Shade background by SLIP-FSM phase (skipped if no FSM data)."""
    if phase.max() < 0:
        return
    uniq = [s for s in sorted(set(phase.tolist())) if s >= 0]
    for s in uniq:
        mask = phase == s
        if not mask.any():
            continue
        # contiguous runs
        idx = np.flatnonzero(mask)
        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in splits:
            if len(run) < 2:
                continue
            t0, t1 = t[run[0]], t[run[-1]]
            ax.axvspan(t0, t1, color=PHASE_COLORS.get(int(s), "#eeeeee"), alpha=0.5, lw=0)
    # phase labels (one per phase, at its midpoint)
    for s in uniq:
        mask = phase == s
        if mask.sum() < 2:
            continue
        tm = float(np.median(t[mask]))
        ax.text(
            tm,
            1.02,
            PHASES[int(s)],
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=6.5,
            color="#444444",
            style="italic",
        )


def _style_ax(ax) -> None:
    """Nature 风格: 衰退网格 + 细脊线 (配色经 validate_palette.js 验证)."""
    ax.grid(alpha=0.2, lw=0.4, color="#bbbbbb")
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def fig_height() -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (stem, label) in zip(axes.flatten(), ALGOS):
        d = load_traj(stem)
        t, z, phase = d["t"], d["base_z"], d["phase"]
        shade_phases(ax, t, phase, t[-1])
        ax.plot(t, z, color=COLORS[label], lw=1.8)
        # 训练中终止 (如纯 PPO 跳后摔倒): 曲线截断, 标 ×
        if bool(np.asarray(d.get("terminated", [False]))[0]):
            ax.plot(t[-1], z[-1], marker="x", ms=8, mew=1.8, color=COLORS[label], zorder=4)
        ax.set_title(f"({chr(97 + list(ALGOS).index((stem, label)))}) {label}", fontsize=10)
        ax.tick_params(labelsize=7)
        _style_ax(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Base Height (m)", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=8)
    out = RESULTS / "paper_fig_trajectory"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"跳跃高度曲线图 -> {out}.png")


def fig_joints() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, layout="constrained")
    for ax, (stem, label) in zip(axes.flatten(), ALGOS):
        d = load_traj(stem)
        t, hip, knee = d["t"], d["hip_pitch"], d["knee"]
        # knee L/R (largest excursion), hip L/R (pitch)
        ax.plot(t, np.degrees(knee[:, 0]), color=COLORS[label], lw=1.4, ls="-", label="knee L")
        ax.plot(t, np.degrees(knee[:, 1]), color=COLORS[label], lw=1.4, ls="--", label="knee R")
        ax.plot(t, np.degrees(hip[:, 0]), color="#555555", lw=1.0, ls="-", label="hip L")
        ax.plot(t, np.degrees(hip[:, 1]), color="#555555", lw=1.0, ls="--", label="hip R")
        ax.set_title(f"({chr(97 + list(ALGOS).index((stem, label)))}) {label}", fontsize=10)
        ax.set_ylim(-90, 90)
        ax.tick_params(labelsize=7)
        _style_ax(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Joint Angle (deg)", fontsize=8)
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=8)
    axes[0, 0].legend(
        fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(1.0, 1.42), frameon=False
    )
    out = RESULTS / "paper_fig_jump_joints"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"关节角度曲线图 -> {out}.png")


def main() -> int:
    fig_height()
    fig_joints()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
