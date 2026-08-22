#!/usr/bin/env python3
"""四算法跳跃对比图 (nature-draw.md v2.1 规范, Okabe-Ito 色盲安全调色板).

图 1  fig_jump_flat_training_2x2_v3  — 训练曲线全景 (return/ep_len/jump reward/policy std)
图 2  fig_jump_flat_final_perf_2x1_v3 — 回合分布 + 恢复成功率置信区间
图 3  fig_jump_flat_traj_2x1_v3      — 相对站立高度 / 垂直速度, SRL-FSM 相位带

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
# 图例/画线顺序: 本文方法在前，随后为三个组件消融。
ALGOS = ["SRL+VMC", "SRL", "PPO+VMC", "PPO"]
ALGO_FULL = {"PPO": "PPO", "PPO+VMC": "PPO+VMC", "SRL": "SRL", "SRL+VMC": "SRL+VMC"}

TRAIN_LOGS = {
    "PPO": "logs/train/jump_flat_final10000.log",
    "PPO+VMC": "logs/train/jump_vmc_final10000.log",
    "SRL": "logs/train/jump_srl_final10000.log",
    "SRL+VMC": "logs/train/jump_srl_vmc_v8e5.log",
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
    "axes.grid": False,
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
        ("reward", "Mean episodic return", True),
        ("ep_len", "Episode length (steps)", True),
        ("jump_height", "Jump-height reward", True),
        ("std", "Policy action std", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FULL, FULL * 0.55))
    data = {a: parse_train_log(TRAIN_LOGS[a]) for a in ALGOS}
    for panel_idx, (ax, (tag, ylabel, smooth)) in enumerate(zip(axes.flatten(), panels)):
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
        panel_label(ax, f"({chr(97 + panel_idx)})", dx=-0.02, dy=1.03)
        if panel_idx >= 2:
            ax.set_xlabel("Training iteration (×10$^3$)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(0, 10)
        ax.yaxis.grid(True, color="#D8D8D8", lw=0.35, alpha=0.55, zorder=0)
        _style_ax(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(ALGOS), frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.88,
                        wspace=0.28, hspace=0.38)
    save(fig, "training", "2x2", "v3", "fig_jump_flat_training_2x2_v3")


# ================= 图 2: 最终性能柱状 2×1 =================

def fig_final_perf() -> None:
    jf = ROOT / "logs" / "pose_data" / "jump_final_metrics_20260817.json"
    raw = json.loads(jf.read_text())
    metrics = {a: {k: np.array([ep[k] for ep in raw[a]]) for k in raw[a][0].keys()}
               for a in raw}
    x = np.arange(len(ALGOS))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, FULL * 0.34), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    colors = [ALGO_COLORS[a] for a in ALGOS]

    # Independent single-jump evaluation (n=20). Show every observation;
    # boxes encode the IQR and the white diamond is the arithmetic mean.
    rng = np.random.default_rng(20260817)
    height_runs = [metrics[a]["jump_height"] for a in ALGOS]
    for xi, (algo, values) in enumerate(zip(ALGOS, height_runs)):
        parts = axes[0].violinplot(values, positions=[xi], widths=0.72,
                                  showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(ALGO_COLORS[algo])
            body.set_edgecolor("none")
            body.set_alpha(0.18)
        axes[0].boxplot(values, positions=[xi], widths=0.20, patch_artist=True,
                        showfliers=False,
                        boxprops={"facecolor": "white", "edgecolor": ALGO_COLORS[algo], "lw": 1.0},
                        whiskerprops={"color": ALGO_COLORS[algo], "lw": 0.8},
                        capprops={"color": ALGO_COLORS[algo], "lw": 0.8},
                        medianprops={"color": "#222222", "lw": 1.0})
        jitter = rng.uniform(-0.105, 0.105, len(values))
        axes[0].scatter(xi + jitter, values, s=9, color=ALGO_COLORS[algo],
                        edgecolor="white", linewidth=0.25, alpha=0.78, zorder=4)
        mean = float(np.mean(values))
        axes[0].scatter([xi], [mean], marker="D", s=20, facecolor="white",
                        edgecolor="#222222", linewidth=0.7, zorder=6)
        axes[0].text(xi, max(values) + 0.010, f"{mean:.3f}", ha="center", fontsize=6.5)
    axes[0].set_ylabel("Jump height (m)", fontsize=9)
    axes[0].set_ylim(0.07, max(max(v) for v in height_runs) + 0.035)
    axes[0].yaxis.grid(True, color="#D8D8D8", lw=0.35, alpha=0.55, zorder=0)

    # Repeated-trigger protocol (n=20): parse episode rows so the plot remains
    # reproducible from the raw evaluation logs rather than hand-entered data.
    repeat_logs = {
        "PPO": ROOT / "logs/pose_data/jump_repeat_20260817/ppo.txt",
        "PPO+VMC": ROOT / "logs/pose_data/jump_repeat_20260817/ppo_vmc.txt",
        "SRL": ROOT / "logs/pose_data/jump_repeat_20260817/srl.txt",
        "SRL+VMC": ROOT / "logs/pose_data/jump_repeat_20260817/srl_vmc.txt",
    }
    row_re = re.compile(r"^ep\S+\s+(True|False)\s*(True|False)\s*(True|False)\s+")
    successes = []
    for algo in ALGOS:
        rows = [row_re.match(line) for line in repeat_logs[algo].read_text().splitlines()]
        rows = [m for m in rows if m]
        if len(rows) != 20:
            raise ValueError(f"expected 20 repeat rows for {algo}, got {len(rows)}")
        successes.append(sum(m.group(2) == "True" for m in rows))

    def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        p = k / n
        den = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        return center - half, center + half

    ypos = np.arange(len(ALGOS))[::-1]
    for y, algo, k in zip(ypos, ALGOS, successes):
        value = k / 20
        lo, hi = wilson_interval(k, 20)
        axes[1].errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt="o",
                         ms=5.2, mfc=ALGO_COLORS[algo], mec="white", mew=0.5,
                         ecolor=ALGO_COLORS[algo], elinewidth=1.2, capsize=3, zorder=4)
        axes[1].text(min(value + 0.012, 1.006), y + 0.16, f"{k}/20",
                     ha="right" if value > 0.98 else "left", fontsize=6.5)
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels([ALGO_FULL[a] for a in ALGOS], fontsize=7)
    axes[1].set_xlabel("Landing-recovery success (95% CI)", fontsize=9)
    axes[1].set_xlim(0.66, 1.015)
    axes[1].set_xticks([0.7, 0.8, 0.9, 1.0])
    axes[1].set_xticklabels(["70", "80", "90", "100"])
    axes[1].xaxis.grid(True, color="#D8D8D8", lw=0.35, alpha=0.55, zorder=0)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([ALGO_FULL[a] for a in ALGOS], fontsize=7)
    for ax in axes:
        _style_ax(ax)
    for ax, lab in zip(axes, "ab"):
        panel_label(ax, f"({lab})")
    save(fig, "final_perf", "2x1", "v3", "fig_jump_flat_final_perf_2x1_v3")


# ================= 图 3: 典型跳跃轨迹 2×1 =================

PHASES = {-1: "idle", 0: "crouch", 1: "thrust", 2: "flight", 3: "landing", 4: "recover"}
PHASE_COLORS = {
    0: "#cfe3f4", 1: "#fde8cd", 2: "#d5efdf", 3: "#f5dad1", 4: "#e8e8e8",
}
TRAJ_STEMS = {"PPO+VMC": "jump_traj_vmc", "SRL": "jump_traj_srl",
              "SRL+VMC": "jump_traj_srlvmc", "PPO": "jump_traj_ppo"}


def _shade_srl_phases(ax, t, phase, dt, show_labels: bool = True) -> None:
    if phase.max() < 0:
        return
    label_runs = []
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
            # Label only the first meaningful contiguous occurrence.  Taking
            # the median over every occurrence made labels overlap when the
            # FSM briefly revisited a state during recovery.
            if not any(item[0] == int(s) for item in label_runs) and len(run) * dt >= 0.06:
                label_runs.append((int(s), run))
    if not show_labels:
        return
    for s, run in sorted(label_runs, key=lambda item: item[1][0]):
        tm = float((t[run[0]] + t[run[-1]]) / 2)
        duration = float(t[run[-1]] - t[run[0]])
        short = duration < 0.20
        ax.text(tm, 0.965, PHASES[s],
                transform=ax.get_xaxis_transform(), ha="center",
                va="top", rotation=90 if short else 0,
                fontsize=4.8 if short else 5.5, color="#444444", style="italic")


def fig_trajectory() -> None:
    # Use the proposed method's first FSM cycle as the phase reference.
    srl = np.load(DATA / "jump_traj_srlvmc.npz")
    phase = np.asarray(srl["phase"])
    active = np.flatnonzero(phase >= 0)
    t0 = float(srl["t"][active[0]]) if len(active) else 0.5
    fig, axes = plt.subplots(1, 2, figsize=(FULL, FULL * 0.34), layout="constrained",
                             sharex=True)
    ax_h, ax_v = axes
    for algo in ALGOS:
        d = np.load(DATA / f"{TRAJ_STEMS[algo]}.npz")
        t = np.array(d["t"]) - t0
        z = np.array(d["base_z"])
        # The terminal step may already contain the simulator reset state.
        # Exclude it to avoid a nonphysical vertical spike at the trace end.
        if bool(np.asarray(d["terminated"]).reshape(-1)[0]) and len(t) > 1:
            t, z = t[:-1], z[:-1]
        pre = z[t < 0]
        standing = float(np.median(pre[-20:])) if len(pre) else float(z[0])
        ax_h.plot(t, z - standing, color=ALGO_COLORS[algo], lw=1.25,
                  label=ALGO_FULL[algo], zorder=3)
        if "linvel" in d.files and d["linvel"].shape[1] >= 3:
            vz = np.array(d["linvel"])[:, 2]
            vz = vz[:len(t)]
            ax_v.plot(t, vz, color=ALGO_COLORS[algo], lw=1.3, label=ALGO_FULL[algo], zorder=3)
        else:
            vz = np.gradient(z, np.array(d["t"])) * 1.0
            ax_v.plot(t, vz, color=ALGO_COLORS[algo], lw=1.3, zorder=3)
    phase_t = np.array(srl["t"]) - t0
    phase = np.array(srl["phase"])
    _shade_srl_phases(ax_h, phase_t, phase, float(srl["ctrl_dt"]), show_labels=True)
    _shade_srl_phases(ax_v, phase_t, phase, float(srl["ctrl_dt"]), show_labels=False)
    panel_label(ax_h, "(a)", dx=0.015, dy=0.955)
    panel_label(ax_v, "(b)", dx=0.015, dy=0.955)
    ax_h.set_ylabel("Vertical displacement (m)", fontsize=9)
    ax_v.set_ylabel("Vertical vel. $v_z$ (m/s)", fontsize=9)
    for ax in (ax_h, ax_v):
        ax.set_xlabel("Time relative to trigger (s)", fontsize=9)
    ax_v.axhline(0, color="#888888", lw=0.5, ls=":", zorder=1)
    for ax in (ax_h, ax_v):
        ax.set_xlim(-0.15, 1.28)
        ax.yaxis.grid(True, color="#D8D8D8", lw=0.35, alpha=0.55, zorder=0)
        _style_ax(ax)
    handles, labels = ax_h.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.025),
               ncol=4, frameon=False, fontsize=7, columnspacing=1.6,
               handlelength=2.0)
    save(fig, "traj", "2x1", "v3", "fig_jump_flat_traj_2x1_v3")


def main() -> int:
    print("图 1: 训练曲线全景 2×2")
    fig_training_metrics()
    print("图 2: 回合分布与恢复成功率 2×1")
    fig_final_perf()
    print("图 3: 典型跳跃轨迹 2×1")
    fig_trajectory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
