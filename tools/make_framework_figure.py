#!/usr/bin/env python3
"""Generate the SRL+VMC framework figure for the wheel-legged jumping paper.

Panel (a): the proposed method -- SLIP-FSM phase prior, PPO residual policy and
the phase-adaptive VMC control layer.
Panel (b): the 2×2 component-ablation matrix used to evaluate the method.

Style follows the nature/dataviz conventions used by make_paper_figures.py:
Okabe-Ito colourblind-safe palette, thin spines, no chartjunk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# SimHei for Chinese labels (falls back to Noto Sans CJK on systems without it)
_CN = "SimHei"
for _p in [ROOT / "latex/two_wheeled_robot_thesis/font/SimHei.ttf"]:
    if _p.exists():
        fm.fontManager.addfont(str(_p))
if "SimHei" not in {f.name for f in fm.fontManager.ttflist}:
    _CN = "Noto Sans CJK JP"
plt.rcParams["font.family"] = _CN
plt.rcParams["axes.unicode_minus"] = False

# Okabe-Ito palette
C = {
    "SRL": "#0072B2",
    "PPO": "#E69F00",
    "PPO+VMC": "#009E73",
    "VMC+SRL": "#D55E00",
    "ink": "#222222",
    "grey": "#888888",
    "box": "#f2f2f2",
    "box2": "#e7eef4",
}


def _box(ax, x, y, w, h, text, fc, ec, fs=9, weight="normal", lw=0.9, ha="center"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        fc=fc,
        ec=ec,
        lw=lw,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=C["ink"], weight=weight
    )
    return p


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.2):
    a = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=12,
        color=color,
        lw=lw,
    )
    ax.add_patch(a)


def panel_b(ax):
    """2×2 component-ablation matrix."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Axis labels
    ax.text(5.0, 9.3, "控制空间", ha="center", fontsize=10, weight="bold", color=C["ink"])
    ax.text(2.6, 9.3, "关节空间（位置控制）", ha="center", fontsize=8.5, color=C["grey"])
    ax.text(7.6, 9.3, "虚拟腿 VMC（力矩控制）", ha="center", fontsize=8.5, color=C["grey"])
    ax.text(
        0.62,
        4.4,
        "参考\n轨迹",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        color=C["ink"],
    )

    # Cell positions
    cells = [
        # (x, y, variant, name, line1, line2, color)
        (1.6, 6.0, "纯PPO", "无相位先验 / 关节PD", "#fdf3e0", C["PPO"]),
        (5.2, 6.0, "PPO+VMC", "无相位先验 / 虚拟腿力控", "#e8f5ee", C["PPO+VMC"]),
        (1.6, 1.2, "SRL", "SLIP-FSM / 关节PD", "#e1edf7", C["SRL"]),
        (5.2, 1.2, "SRL+VMC（本文）", "SLIP-FSM / 虚拟腿力控", "#f8e6dd", C["VMC+SRL"]),
    ]
    for x, y, name, sub, fc, ec in cells:
        _box(ax, x, y, 3.3, 1.5, "", fc, ec, lw=1.1)
        ax.text(
            x + 1.65, y + 1.05, name, ha="center", va="center", fontsize=10, weight="bold", color=ec
        )
        ax.text(x + 1.65, y + 0.42, sub, ha="center", va="center", fontsize=8, color="#444444")

    # Row labels
    ax.text(1.0, 7.35, "无参考", ha="center", fontsize=9, color=C["grey"])
    ax.text(1.0, 2.55, "SLIP-FSM 参考", ha="center", fontsize=9, color=C["grey"])


def panel_a(ax):
    """Proposed SRL+VMC data-flow pipeline."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, 0.2, 7.0, 2.7, 1.6, "SLIP-FSM\n六状态相位先验", C["box"], C["SRL"], fs=9, weight="bold")
    _box(ax, 0.2, 4.8, 2.7, 1.5, "本体观测 $o_t$\n跳跃指令 $j_{cmd}$", C["box"], "#777777", fs=8.5)
    _box(ax, 3.6, 7.0, 2.7, 1.6, "PPO 残差策略\n$\\Delta u_t=\\pi(o_t,s_t)$", C["box"], "#555555", fs=9, weight="bold")
    _box(ax, 3.6, 4.6, 2.7, 1.4, "虚拟腿目标融合\n$u_t=u_{FSM}+g\\Delta u_t$", C["box2"], "#555555", fs=8.5)
    _box(ax, 6.9, 5.8, 2.7, 2.2, "相位自适应 VMC\n变刚度 / 变阻尼\n雅可比力矩映射", C["box"], C["VMC+SRL"], fs=8.5, weight="bold")

    # Robot
    _box(ax, 6.9, 1.2, 2.7, 1.6, "xqrobotwl\n两轮足机器人", C["box"], C["ink"], fs=9, weight="bold")

    # Arrows
    _arrow(ax, 2.9, 7.8, 3.6, 7.8)
    _arrow(ax, 2.9, 5.7, 3.6, 5.4)  # obs -> policy (dashed below)
    _arrow(ax, 4.9, 7.0, 4.9, 6.0)  # policy -> fusion
    _arrow(ax, 6.3, 5.3, 6.9, 6.4)
    _arrow(ax, 8.2, 5.8, 8.2, 2.8)  # control -> robot
    # feedback arrow
    a = FancyArrowPatch(
        (8.2, 1.2), (8.2, 0.4), arrowstyle="-|>", mutation_scale=12, color=C["grey"], lw=1.0
    )
    ax.add_patch(a)
    ax.text(8.75, 0.55, "状态反馈", fontsize=7.5, color=C["grey"])


def main() -> int:
    out = ROOT / "latex/Wheeled-SRL-Jumping/figures/framework.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
    panel_a(axes[0])
    panel_b(axes[1])
    axes[0].set_title("(a) SRL+VMC 控制框架", fontsize=10, color=C["ink"])
    axes[1].set_title("(b) 组件消融设计", fontsize=10, color=C["ink"])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"框架图 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
