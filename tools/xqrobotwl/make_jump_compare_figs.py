#!/usr/bin/env python3
"""四算法跳跃对比图 (nature-draw.md v2.1 规范, Okabe-Ito 色盲安全调色板).

图 1  fig_jump_flat_training_2x2_v2  — 训练曲线全景 (reward/ep_len/jump_height/action_std)
图 2  fig_jump_flat_final_perf_2x1_v2 — 最终性能柱状 (跳高 / 站姿|gyro|), mean±std
图 3  fig_jump_flat_traj_2x1_v2      — 典型跳跃轨迹 (base_z / 垂直速度, SRL-FSM 相位带)

数据源:
  训练曲线: logs/train/jump_full_{ppo,vmc,srl,srl_vmc}.log (官方 10000 轮对比训练)
  最终性能: logs/pose_data/jump_final_metrics.json (collect_jump_metrics.py 重复评估)
  轨迹:     latex/Wheeled-SRL-Jumping/data/jump_traj_{ppo,vmc,srl,srlvmc}.npz (最终模型)

输出: picture/paper/jump/<内容>/<版式>/<版本>/  (PDF 矢量 + PNG 600dpi)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIG_BASE = ROOT / "picture" / "paper" / "jump"
DATA = ROOT / "latex" / "Wheeled-SRL-Jumping" / "data"

# ---- 尺寸 (nature-draw.md §一) ----
COL = 3.5
FULL = 7.25
GOLDEN = (5**0.5 - 1) / 2

# ---- Okabe-Ito 色盲安全调色板 (nature-draw.md §四) ----
ALGO_COLORS = {
    "PPO": "#0072B2",      # 深蓝 (基准算法)
    "PPO+VMC": "#D55E00",  # 橙
    "SRL": "#009E73",      # 绿
    "SRL+VMC": "#CC79A7",  # 紫
}
# 图例/画线顺序: 按最终性能(跳高)从高到低
ALGOS = ["PPO+VMC", "SRL", "SRL+VMC", "PPO"]
ALGO_FULL = {"PPO": "PPO", "PPO+VMC": "PPO+VMC", "SRL": "SRL", "SRL+VMC": "SRL+VMC"}

TRAIN_LOGS = {
    "PPO": "logs/train/jump_full_ppo.log",
    "PPO+VMC": "logs/train/jump_full_vmc.log",
    "SRL": "logs/train/jump_full_srl.log",
    "SRL+VMC": "logs/train/jump_full_srl_vmc.log",
}

# ---- 全局样式 (nature-draw.md §二/§三) ----
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.fontset": "stix",
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.3,
    "grid.alpha": 0.25,
    "grid.color": "#CCCCCC",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ================= 工具函数 =================

def ema(values: np.ndarray, alpha: float = 0.02) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    acc = float(values[0])
    for i, v in enumerate(values):
        acc = (1 - alpha) * acc + alpha * float(v)
        out[i] = acc
    return out


def downsample(steps, values, max_points: int = 1200):
    if len(steps) <= max_points:
        return steps, values
    idx = np.linspace(0, len(steps) - 1, max_points).astype(int)
    return steps[idx], values[idx]


def panel_label(ax, text: str, dx: float = 0.0, dy: float = 1.02) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9, fontweight="bold")


def _style_ax(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
    for s in ax.spines.values():
        s.set_linewidth(0.7)


def save(fig, content: str, layout: str, version: str, name: str) -> None:
    out_dir = FIG_BASE / content / layout / version
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_dir}/{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.png", format="png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_dir}/{name}.pdf + .png")


# ================= 数据解析 =================

def parse_train_log(path: str) -> dict[str, np.ndarray]:
    """Parse a train log → {tag: values[iter]} for iter-aligned tags."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", Path(path).read_text(errors="replace"))
    lines = clean.splitlines()
    iters, reward, ep_len, std, jh = [], [], [], [], []
    cur_iter = 0
    for ln in lines:
        s = ln.strip()
        m = re.match(r"Learning iteration (\d+)/(\d+)", s)
        if m:
            cur_iter = int(m.group(1))
        m = re.match(r"Mean reward: (-?[\d.eE+]+)", s)
        if m:
            reward.append((cur_iter, float(m.group(1))))
        m = re.match(r"Mean episode length: ([\d.eE+]+)", s)
        if m:
            ep_len.append((cur_iter, float(m.group(1))))
        m = re.match(r"Mean action std: ([\d.eE+]+)", s)
        if m:
            std.append((cur_iter, float(m.group(1))))
        m = re.match(r"reward/jump_height: (-?[\d.eE+]+)", s)
        if m:
            jh.append((cur_iter, float(m.group(1))))
    def _arr(pairs):
        if not pairs:
            return np.array([]), np.array([])
        a = np.array(pairs)
        return a[:, 0], a[:, 1]
    return {"reward": _arr(reward), "ep_len": _arr(ep_len),
            "std": _arr(std), "jump_height": _arr(jh)}


# ================= 图 1: 训练曲线全景 2×2 =================

def fig_training_metrics() -> None:
    panels = [
        ("reward", "Mean Reward", True),
        ("ep_len", "Episode Length", True),
        ("jump_height", "Jump Height Reward", True),
        ("std", "Action Std", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.62), layout="constrained")
    data = {a: parse_train_log(TRAIN_LOGS[a]) for a in ALGOS}
    for ax, (tag, title, smooth) in zip(axes.flatten(), panels):
        for algo in ALGOS:
            steps, values = data[algo][tag]
            if len(values) == 0:
                continue
            s, v = downsample(steps, values)
            ax.plot(s / 1000.0, v, color=ALGO_COLORS[algo], lw=0.4,
                    alpha=0.22, zorder=1)
            sv = ema(values) if smooth else values
            s2, v2 = downsample(steps, sv)
            ax.plot(s2 / 1000.0, v2, label=ALGO_FULL[algo], color=ALGO_COLORS[algo],
                    lw=1.4, zorder=3)
        panel_label(ax, f"({chr(97 + panels.index((tag, title, smooth)))})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training iteration (×10$^3$)", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_xlim(0, 10)
        _style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(ALGOS), frameon=False, fontsize=7)
    fig.tight_layout(pad=0.6, rect=(0, 0.06, 1, 1))
    save(fig, "training", "2x2", "v2", "fig_jump_flat_training_2x2_v2")


# ================= 图 2: 最终性能柱状 2×1 =================

def fig_final_perf() -> None:
    jf = ROOT / "logs" / "pose_data" / "jump_final_metrics.json"
    raw = json.loads(jf.read_text())
    metrics = {a: {k: np.array([ep[k] for ep in raw[a]]) for k in raw[a][0].keys()}
               for a in raw}
    x = np.arange(len(ALGOS))
    panels = [("jump_height", "Jump Height (m)", True),
              ("stand_gyro", "Standing $|\\omega|$ (rad/s)", False)]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, FULL * 0.34), layout="constrained")
    for ax, (key, ylabel, higher_better) in zip(axes, panels):
        vals = [metrics[a][key].mean() for a in ALGOS]
        stds = [metrics[a][key].std() for a in ALGOS]
        colors = [ALGO_COLORS[a] for a in ALGOS]
        ax.bar(x, vals, 0.6, color=colors, yerr=stds, capsize=4,
               error_kw={"elinewidth": 0.8, "ecolor": "#444444"}, zorder=3)
        for xi, v, s in zip(x, vals, stds):
            ax.text(xi, v + s + 0.01, f"{v:.2f}", ha="center", fontsize=6, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_FULL[a] for a in ALGOS], fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.35)
        _style_ax(ax)
    for ax, lab in zip(axes, "ab"):
        panel_label(ax, f"({lab})")
    save(fig, "final_perf", "2x1", "v2", "fig_jump_flat_final_perf_2x1_v2")


# ================= 图 3: 典型跳跃轨迹 2×1 =================

PHASES = {-1: "idle", 0: "crouch", 1: "thrust", 2: "flight", 3: "landing", 4: "recover"}
PHASE_COLORS = {
    0: "#cfe3f4", 1: "#fde8cd", 2: "#d5efdf", 3: "#f5dad1", 4: "#e8e8e8",
}
TRAJ_STEMS = {"PPO+VMC": "jump_traj_vmc", "SRL": "jump_traj_srl",
              "SRL+VMC": "jump_traj_srlvmc", "PPO": "jump_traj_ppo"}


def _shade_srl_phases(ax, t, phase, dt) -> None:
    if phase.max() < 0:
        return
    for s in sorted(set(phase.tolist())):
        if s < 0:
            continue
        mask = phase == s
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for run in splits:
            if len(run) < 2:
                continue
            ax.axvspan(t[run[0]], t[run[-1]],
                       color=PHASE_COLORS.get(int(s), "#eeeeee"), alpha=0.4, lw=0, zorder=0)
    for s in sorted(set(phase.tolist())):
        if s < 0:
            continue
        mask = phase == s
        if mask.sum() < 2:
            continue
        tm = float(np.median(t[mask]))
        ax.text(tm, 1.02, PHASES[int(s)], transform=ax.get_xaxis_transform(),
                ha="center", fontsize=5.5, color="#444444", style="italic")


def fig_trajectory() -> None:
    # 用 SRL 的相位作参考带 (SLIP-FSM 规范跳跃时序)
    srl = np.load(DATA / "jump_traj_srl.npz")
    t0 = float(srl["t"][39])  # settle=40, 触发起点
    fig, axes = plt.subplots(2, 1, figsize=(COL, COL * 1.3), layout="constrained",
                             sharex=True)
    ax_h, ax_v = axes
    for algo in ALGOS:
        d = np.load(DATA / f"{TRAJ_STEMS[algo]}.npz")
        t = np.array(d["t"]) - t0
        z = np.array(d["base_z"])
        ax_h.plot(t, z, color=ALGO_COLORS[algo], lw=1.3, label=ALGO_FULL[algo], zorder=3)
        if "linvel" in d.files and d["linvel"].shape[1] >= 3:
            vz = np.array(d["linvel"])[:, 2]
            ax_v.plot(t, vz, color=ALGO_COLORS[algo], lw=1.3, label=ALGO_FULL[algo], zorder=3)
        else:
            vz = np.gradient(z, np.array(d["t"])) * 1.0
            ax_v.plot(t, vz, color=ALGO_COLORS[algo], lw=1.3, zorder=3)
    _shade_srl_phases(ax_h, np.array(srl["t"]) - t0, np.array(srl["phase"]), float(srl["ctrl_dt"]))
    panel_label(ax_h, "(a)")
    panel_label(ax_v, "(b)")
    ax_h.set_ylabel("Base height (m)", fontsize=9)
    ax_v.set_ylabel("Vertical vel. $v_z$ (m/s)", fontsize=9)
    ax_v.set_xlabel("Time relative to trigger (s)", fontsize=9)
    ax_v.axhline(0, color="#888888", lw=0.5, ls=":", zorder=1)
    for ax in (ax_h, ax_v):
        _style_ax(ax)
    handles, labels = ax_h.get_legend_handles_labels()
    ax_h.legend(handles, labels, loc="upper left", frameon=False, fontsize=7)
    save(fig, "traj", "2x1", "v2", "fig_jump_flat_traj_2x1_v2")


def main() -> int:
    print("图 1: 训练曲线全景 2×2")
    fig_training_metrics()
    print("图 2: 最终性能柱状 2×1")
    fig_final_perf()
    print("图 3: 典型跳跃轨迹 2×1")
    fig_trajectory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
