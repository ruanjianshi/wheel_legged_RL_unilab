#!/usr/bin/env python3
"""Wheeled-SRL 论文出图脚本 v2.0 — 严格遵循 nature-draw.md 规范.

规范要点 (v2.0):
- 尺寸: 单栏 3.39in / 跨栏 7.09in (本文按实际 \columnwidth≈6.3in 渲染, 字体字号精确)
- 字体: 无衬线 (Helvetica/Arial/Liberation Sans), 轴标签 9pt, 刻度 7pt, 图例 7pt
- 线条: 数据 1.5pt, 坐标轴 0.7pt 仅左下可见+外移 5pt, 刻度向内, 网格仅 y 轴 (#ADB5BD, a=0.2)
- 配色: IBM 调色板 — PPO #4263EB / SRL #40C057 / PPO+VMC #FA5252 / SRL+VMC #7950F2
- 训练曲线: EMA α=0.02 平滑主曲线, 降采样 ≤1200 点, x 轴用 k 单位
- 子图编号: (a)/(b)... 左上角粗体 9pt; 图例无边框, 图下居中
- 输出: PDF 矢量 (fonttype=42) + PNG 600dpi

生成 4 张数据图 (命名规范 §九: fig_[步态]_[任务]_[内容]_[版式]_[版本]):
  1. fig_jump_flat_training_2x3_v2  — 训练指标 2×3 (奖励/回合/跳高奖励/软着陆/动作std/FPS)
  2. fig_jump_flat_final_perf_3x1_v2 — 最终性能柱状 (跳高/腾空率/存活率)
  3. fig_jump_flat_traj_2x2_v2       — 跳跃高度轨迹 2×2 (含 FSM 相位色带)
  4. fig_jump_flat_joints_2x2_v2     — 腿部关节角 2×2
另按 IBM 配色重生成 framework.pdf/png (2×2 设计 + 控制流水线).

用法:
  uv run python tools/make_paper_figures_v2.py
  uv run python tools/make_paper_figures_v2.py --no-framework   # 只出数据图
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

ROOT = Path(__file__).resolve().parent.parent  # 仓库根 (本文件位于 tools/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG = ROOT / "latex" / "Wheeled-SRL-Jumping" / "figures"
DATA = ROOT / "latex" / "Wheeled-SRL-Jumping" / "data"
COL = 3.39          # 单栏宽度 (in)
FULL = 7.09         # 跨栏宽度 (in)
TEXTW = 6.30        # 本论文 \columnwidth ≈ 16cm (A4 去 2.5cm 边距)
GOLDEN = (5**0.5 - 1) / 2

# ---- IBM 现代科学调色板 (nature-draw.md §四) ----
ALGO_COLORS = {
    "PPO": "#4263EB",      # 深蓝 (纯PPO, 基准)
    "PPO+VMC": "#FA5252",  # 深红
    "SRL": "#40C057",      # 深绿 (最优方法)
    "SRL+VMC": "#7950F2",  # 紫
}
# 按最终性能 (跳高) 从高到低: SRL 0.547 > SRL+VMC 0.352 > PPO 0.264 > PPO+VMC 0.175
ALGOS = ["SRL", "SRL+VMC", "PPO", "PPO+VMC"]

# ---- 全局样式 (nature-draw.md §二/§三) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.fontset": "stixsans",
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.3,
    "grid.alpha": 0.2,
    "grid.color": "#ADB5BD",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ================= 工具函数 =================

def ema(values: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    """EMA 平滑 (nature-draw.md §五): y_i = (1-α)y_{i-1} + α x_i, α=0.02."""
    out = np.empty_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    acc = float(values[0])
    for i, v in enumerate(values):
        acc = (1 - alpha) * acc + alpha * float(v)
        out[i] = acc
    return out


def downsample(steps, values, max_points: int = 1200):
    """降采样到 ≤ max_points 点 (nature-draw.md §五)."""
    if len(steps) <= max_points:
        return steps, values
    idx = np.linspace(0, len(steps) - 1, max_points).astype(int)
    return steps[idx], values[idx]


def panel_label(ax, text: str, dx: float = 0.0, dy: float = 1.02) -> None:
    """子图编号: 左上角 (0, 1.02), axes 坐标系, 粗体 9pt."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9, fontweight="bold")


def _style_ax(ax) -> None:
    """Nature 风格: 仅左下 spine 可见+外移 5pt, 细脊线, 刻度向内, 网格仅 y 轴."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    for s in ax.spines.values():
        s.set_linewidth(0.7)


def save(fig, name: str) -> None:
    """输出: PDF 矢量 + PNG 600dpi (nature-draw.md §七)."""
    fig.savefig(f"{FIG}/{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.pdf + .png (600dpi)")


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
    """4 变体训练 run (与论文表 tab:baseline / four_algo_comparison.json 一致)."""
    base = ROOT / "logs" / "rsl_rl_ppo"
    cfg = {
        "SRL": ("XqRobotWLJumpSRLFlat", "2026-08-06_01-16-20_mujoco"),
        "PPO": ("XqRobotWLJumpFlat", "2026-08-09_01-21-11_mujoco"),  # v12 launch_rise
        "PPO+VMC": ("XqRobotWLJumpVMC", "2026-08-09_01-21-12_mujoco"),
        "SRL+VMC": ("XqRobotWLJumpSRLVMC", "2026-08-08_01-05-51_mujoco"),
    }
    found = {}
    for algo, (task, run) in cfg.items():
        d = base / task / run
        if d.exists():
            found[algo] = str(d)
        else:
            print(f"  WARN: {algo} 训练目录缺失 {d}")
    return found


def _plot_curves(ax, dirs, tag, alpha=0.02, smooth_it=True) -> None:
    """四算法同色并排画一条曲线 (EMA α=0.02, 降采样≤1200)."""
    for algo in ALGOS:
        if algo not in dirs:
            continue
        data = load_tb(dirs[algo], [tag]).get(tag)
        if data is None or len(data["values"]) == 0:
            print(f"  WARN: {algo} 无 {tag}")
            continue
        steps, values = data["steps"], data["values"]
        if smooth_it:
            values = ema(values, alpha)
        s, v = downsample(steps, values)
        ax.plot(s / 1000.0, v, color=ALGO_COLORS[algo], lw=1.5, label=algo, zorder=3)


def comparison_data() -> tuple[list[str], list[float], list[float], list[float]]:
    """从 four_algo_comparison.json 读取 4 变体跳高/腾空率/存活率."""
    jf = DATA / "four_algo_comparison.json"
    data = json.load(open(jf))["results"]
    keys = {
        "SRL": "XqRobotWLJumpSRLFlat",
        "SRL+VMC": "XqRobotWLJumpSRLVMC",
        "PPO": "XqRobotWLJumpFlat",
        "PPO+VMC": "XqRobotWLJumpVMC",
    }
    h = [data[k]["jump_height"] for k in keys.values()]
    a = [data[k]["air_frac"] for k in keys.values()]
    s = [data[k]["survival"] for k in keys.values()]
    return list(keys), h, a, s


def load_traj(stem: str) -> dict:
    d = np.load(DATA / f"{stem}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


# ================= 图 1: 训练指标 2×3 =================

def fig_training_metrics(dirs) -> None:
    panels = [
        ("Train/mean_reward", "Mean Reward", True),
        ("Train/mean_episode_length", "Episode Length", True),
        ("reward/jump_height", "Jump Height Reward", True),
        ("reward/landing_soft", "Landing Soft Reward", True),
        ("Policy/mean_std", "Action Std", False),
        ("Perf/total_fps", "Training FPS", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, TEXTW * 0.62), layout="constrained")
    for ax, (tag, title, smooth_it) in zip(axes.flatten(), panels):
        _plot_curves(ax, dirs, tag, smooth_it=smooth_it)
        panel_label(ax, f"({chr(97 + panels.index((tag, title, smooth_it)))})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training iteration (×10$^3$)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        _style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.03),
        ncol=4, frameon=False, fontsize=7,
    )
    fig.tight_layout(pad=0.6, rect=(0, 0.06, 1, 1))
    save(fig, "fig_jump_flat_training_2x3_v2")


# ================= 图 2: 最终性能柱状 (跳高/腾空率/存活率) =================

def fig_final_perf() -> None:
    algos, h, a, s = comparison_data()
    groups = [("Jump Height (m)", h), ("Air Fraction", a), ("Survival Rate", s)]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, TEXTW * 0.32), layout="constrained")
    x = np.arange(len(algos))
    for ax, (ylabel, vals) in zip(axes, groups):
        ax.bar(x, vals, width=0.6, color=[ALGO_COLORS[al] for al in algos],
               edgecolor="white", zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, max(max(vals) * 1.25, 1.0))
        _style_ax(ax)
    for ax, lab in zip(axes, "abc"):
        panel_label(ax, f"({lab})")
    save(fig, "fig_jump_flat_final_perf_3x1_v2")


# ================= 图 3: 跳跃高度轨迹 2×2 (含 FSM 相位) =================

PHASES = {-1: "idle", 0: "crouch", 1: "thrust", 2: "flight", 3: "landing", 4: "recover"}
PHASE_COLORS = {
    -1: "#dddddd",
    0: "#cfe3f4",
    1: "#fde8cd",
    2: "#d5efdf",
    3: "#f5dad1",
    4: "#e8e8e8",
}
TRAJ_STEMS = ["jump_traj_srl", "jump_traj_srlvmc", "jump_traj_ppo", "jump_traj_vmc"]


def shade_phases(ax, t, phase) -> None:
    if phase.max() < 0:
        return
    uniq = [s for s in sorted(set(phase.tolist())) if s >= 0]
    for s in uniq:
        mask = phase == s
        if not mask.any():
            continue
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


def fig_trajectory() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, TEXTW * 0.62), sharex=True, sharey=True,
                             layout="constrained")
    for ax, stem in zip(axes.flatten(), TRAJ_STEMS):
        algo = {"jump_traj_srl": "SRL", "jump_traj_srlvmc": "SRL+VMC",
                "jump_traj_ppo": "PPO", "jump_traj_vmc": "PPO+VMC"}[stem]
        d = load_traj(stem)
        t, z, phase = d["t"], d["base_z"], d["phase"]
        shade_phases(ax, t, phase)
        ax.plot(t, z, color=ALGO_COLORS[algo], lw=1.5, zorder=3)
        ax.set_title(algo, fontsize=9)
        panel_label(ax, f"({chr(97 + TRAJ_STEMS.index(stem))})")
        ax.tick_params(labelsize=7)
        _style_ax(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Base Height (m)", fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=9)
    save(fig, "fig_jump_flat_traj_2x2_v2")


# ================= 图 4: 腿部关节角 2×2 =================

def fig_joints() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(TEXTW, TEXTW * 0.62), sharex=True, layout="constrained")
    for ax, stem in zip(axes.flatten(), TRAJ_STEMS):
        algo = {"jump_traj_srl": "SRL", "jump_traj_srlvmc": "SRL+VMC",
                "jump_traj_ppo": "PPO", "jump_traj_vmc": "PPO+VMC"}[stem]
        d = load_traj(stem)
        t, hip, knee = d["t"], d["hip_pitch"], d["knee"]
        ax.plot(t, np.degrees(knee[:, 0]), color=ALGO_COLORS[algo], lw=1.4, ls="-", label="knee L")
        ax.plot(t, np.degrees(knee[:, 1]), color=ALGO_COLORS[algo], lw=1.4, ls="--", label="knee R")
        ax.plot(t, np.degrees(hip[:, 0]), color="#555555", lw=1.0, ls="-", label="hip L")
        ax.plot(t, np.degrees(hip[:, 1]), color="#555555", lw=1.0, ls="--", label="hip R")
        ax.set_title(algo, fontsize=9)
        ax.set_ylim(-90, 90)
        panel_label(ax, f"({chr(97 + TRAJ_STEMS.index(stem))})")
        ax.tick_params(labelsize=7)
        _style_ax(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Joint Angle (deg)", fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)", fontsize=9)
    axes[0, 0].legend(fontsize=7, ncol=4, loc="upper center",
                      bbox_to_anchor=(1.0, 1.42), frameon=False)
    save(fig, "fig_jump_flat_joints_2x2_v2")


# ================= 图 5: framework (IBM 配色) =================

def fig_framework() -> None:
    """按 nature-draw.md 配色重生成 framework.pdf/png (2×2 设计 + 控制流水线)."""
    from matplotlib.font_manager import fontManager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    # 中文字体回退: 优先 SimHei, 否则 Noto Sans CJK
    _CN = "SimHei"
    for _p in [ROOT / "latex/two_wheeled_robot_thesis/font/SimHei.ttf"]:
        if _p.exists():
            fontManager.addfont(str(_p))
    available = {f.name for f in fontManager.ttflist}
    if "SimHei" not in available:
        _CN = "Noto Sans CJK JP"
    if _CN not in available:
        _CN = "Noto Sans CJK HK"
    _old = plt.rcParams["font.family"]
    plt.rcParams["font.family"] = [_CN, "Liberation Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    C = {
        "SRL": ALGO_COLORS["SRL"],
        "PPO": ALGO_COLORS["PPO"],
        "PPO+VMC": ALGO_COLORS["PPO+VMC"],
        "SRL+VMC": ALGO_COLORS["SRL+VMC"],
        "ink": "#222222",
        "grey": "#888888",
        "box": "#f2f2f2",
        "box2": "#e7eef4",
    }

    def _box(ax, x, y, w, h, text, fc, ec, fs=9, weight="normal", lw=0.9, ha="center"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                           fc=fc, ec=ec, lw=lw)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha=ha, va="center", fontsize=fs,
                color=C["ink"], weight=weight)
        return p

    def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=12, color=color, lw=lw))

    def panel_a(ax):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        ax.text(5.0, 9.3, "控制空间", ha="center", fontsize=10, weight="bold", color=C["ink"])
        ax.text(2.6, 9.3, "关节空间（位置控制）", ha="center", fontsize=8.5, color=C["grey"])
        ax.text(7.6, 9.3, "虚拟腿 VMC（力矩控制）", ha="center", fontsize=8.5, color=C["grey"])
        ax.text(0.62, 4.4, "参考\n轨迹", ha="center", va="center", fontsize=10,
                weight="bold", color=C["ink"])
        cells = [
            (1.6, 6.0, "纯PPO", "h=0.264 m  存活 35%", "#fdf3e0", C["PPO"]),
            (5.2, 6.0, "PPO+VMC", "h=0.175 m  存活 95%", "#fbe9e9", C["PPO+VMC"]),
            (1.6, 1.2, "SRL（最优）", "h=0.547 m  存活 100%", "#e6f4e9", C["SRL"]),
            (5.2, 1.2, "SRL+VMC", "h=0.352 m  存活 80%", "#f0e9fb", C["SRL+VMC"]),
        ]
        for x, y, name, sub, fc, ec in cells:
            _box(ax, x, y, 3.3, 1.5, "", fc, ec, lw=1.1)
            ax.text(x + 1.65, y + 1.05, name, ha="center", va="center", fontsize=10,
                    weight="bold", color=ec)
            ax.text(x + 1.65, y + 0.42, sub, ha="center", va="center", fontsize=8, color="#444444")
        ax.text(1.0, 7.35, "无参考", ha="center", fontsize=9, color=C["grey"])
        ax.text(1.0, 2.55, "SLIP-FSM 参考", ha="center", fontsize=9, color=C["grey"])

    def panel_b(ax):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        _box(ax, 0.3, 7.0, 2.6, 1.6, "SLIP-FSM\n六状态相位参考", C["box"], C["SRL"],
             fs=9, weight="bold")
        _box(ax, 0.3, 5.0, 2.6, 1.4, "指令 $j_{cmd}$\n观测 $o_t$", C["box"], "#777777", fs=8.5)
        _box(ax, 3.6, 7.0, 2.6, 1.6, "PPO 策略 $\\pi$", C["box"], "#555555", fs=9, weight="bold")
        _box(ax, 3.6, 4.6, 2.6, 1.4, "前馈+残差\na = g·a_ff + a_π", C["box2"], "#555555", fs=8.5)
        _box(ax, 6.9, 5.8, 2.6, 2.2, "控制层\n关节位置PD\n或 VMC 雅可比力矩", C["box"],
             "#555555", fs=8.5)
        _box(ax, 6.9, 1.2, 2.6, 1.6, "xqrobotwl\n两轮足机器人", C["box"], C["ink"], fs=9,
             weight="bold")
        _arrow(ax, 2.9, 7.8, 3.6, 7.8)
        _arrow(ax, 2.9, 5.7, 3.6, 5.4)
        _arrow(ax, 4.9, 7.0, 4.9, 6.0)
        _arrow(ax, 6.2, 5.3, 6.9, 6.4)
        _arrow(ax, 8.2, 5.8, 8.2, 2.8)
        ax.add_patch(FancyArrowPatch((8.2, 1.2), (8.2, 0.4), arrowstyle="-|>",
                                     mutation_scale=12, color=C["grey"], lw=1.0))
        ax.text(8.75, 0.55, "状态反馈", fontsize=7.5, color=C["grey"])

    fig, axes = plt.subplots(1, 2, figsize=(TEXTW, TEXTW * 0.42), layout="constrained")
    panel_a(axes[0]); panel_b(axes[1])
    axes[0].set_title("(a) 2×2 对照设计", fontsize=10, color=C["ink"])
    axes[1].set_title("(b) 共享控制流水线", fontsize=10, color=C["ink"])
    fig.savefig(f"{FIG}/framework.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/framework.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcParams["font.family"] = _old
    print("  ✓ framework.pdf + .png (300dpi, IBM 配色)")


# ================= 主入口 =================

def main() -> int:
    ap = argparse.ArgumentParser(description="Wheeled-SRL 论文出图 v2.0 (nature-draw 规范)")
    ap.add_argument("--no-framework", action="store_true", help="跳过 framework 重生成")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    dirs = run_dirs()
    print(f"训练目录: {len(dirs)}/4, EMA α=0.02, 输出 → {FIG}")

    print("[1/4] 训练指标 2×3 ...")
    fig_training_metrics(dirs)
    print("[2/4] 最终性能柱状 (跳高/腾空率/存活率) ...")
    fig_final_perf()
    print("[3/4] 跳跃高度轨迹 2×2 ...")
    fig_trajectory()
    print("[4/4] 腿部关节角 2×2 ...")
    fig_joints()
    if not args.no_framework:
        print("[5/5] framework (IBM 配色) ...")
        fig_framework()
    print("完成. 请同步更新 main.tex 引用为 PDF 矢量图.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
