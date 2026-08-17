#!/usr/bin/env python3
"""四算法跳跃高度曲线对比图.

读 record_jump_trajectory.py 记录的 .npz, 以触发时刻 t=0 对齐,
画出四算法 base_z 随时间的变化曲线 + 峰值跳高标注。

Usage:
    uv run tools/xqrobotwl/plot_4algo_height_curve.py --out video/jump/4algo_height_curves.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="video/jump/4algo_height_curves.png")
    p.add_argument("--files", default="", help="逗号分隔 npz 文件; 空=用 logs/jump_traj/*.npz")
    p.add_argument("--settle", type=float, default=0.8, help="触发时刻 (s), 对齐零点")
    args = p.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 注册系统 Noto CJK 字体 (中文字标)
    for f in Path("/usr/share/fonts/opentype/noto").glob("Noto*CJK*.ttc"):
        try:
            font_manager.fontManager.addfont(str(f))
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    if args.files:
        files = [Path(f) for f in args.files.split(",")]
    else:
        files = sorted(Path("logs/jump_traj").glob("XqRobotWLJump*.npz"))

    # 算法短名 + 颜色 + 顺序 (按最终性能排)
    names = {
        "XqRobotWLJumpFlat": ("纯PPO", "#c0392b"),
        "XqRobotWLJumpVMC": ("PPO+VMC", "#d68910"),
        "XqRobotWLJumpSRLFlat": ("SRL", "#16a085"),
        "XqRobotWLJumpSRLVMC": ("SRL+VMC (v8e5)", "#2e86de"),
    }

    fig, ax = plt.subplots(figsize=(11, 6.2))
    peaks = []
    settle_peaks = []
    for f in files:
        d = np.load(f)
        key = Path(f).stem
        label, color = names.get(key, (key, "#333333"))
        t = np.asarray(d["t"]) - args.settle  # t=0 = 触发时刻
        z = np.asarray(d["base_z"])
        ax.plot(t, z, color=color, lw=2.2, label=label, zorder=3)
        stand = float(np.median(z[t < 0]))
        # 真实跳跃峰值 = 触发后 0~2.5s 窗口内的最大高度 (主跳的峰)。
        # 触发前是落地缓冲波动 (纯PPO/PPO+VMC 站立不稳); 触发后 >2.5s 的
        # 二次弹跳 (SRL+VMC 2.8s) / 晚期抖动 (PPO+VMC 3.3s) 都不算。
        post_mask = (t >= 0) & (t <= 2.5)
        tz, zz = t[post_mask], z[post_mask]
        pre = z[t < 0]
        pk_i = int(np.argmax(zz)) if zz.size else 0
        pk = float(zz[pk_i])
        pk_t = float(tz[pk_i])
        peaks.append((label, pk - stand, pk, pk_t, color))
        # 峰值标注
        ax.annotate(
            f"{label}  +{pk - stand:.2f}m",
            xy=(pk_t, pk),
            xytext=(pk_t + 0.12, pk + 0.015),
            fontsize=9,
            color=color,
            ha="left",
            fontweight="bold",
        )
        if pre.size:
            settle_peaks.append((label, float(np.max(pre)) - stand))

    # 触发时刻竖线 + 站立基线
    ax.axvline(0.0, color="#7f8c8d", ls="--", lw=1.2)
    ax.text(0.0, ax.get_ylim()[1] * 0.97, "触发", fontsize=9, color="#7f8c8d", ha="center")

    ax.set_xlabel("时间 (s, 0 = 跳跃触发)", fontsize=12)
    ax.set_ylabel("机身高度 base_z (m)", fontsize=12)
    ax.set_title("四算法平地跳跃高度变化对比 (SRL+VMC v8e4)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.3, lw=0.6)
    ax.set_xlim(-0.8, None)

    # 顶部峰值汇总表
    peaks.sort(key=lambda x: -x[1])
    txt = "\n".join(f"{l}: 跳高 {h:.3f}m" for l, h, *_ in peaks)
    ax.text(
        0.985, 0.02, txt, transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"图已保存 -> {out}")
    print("\n触发后峰值跳高 (单次轨迹):")
    for l, h, pk, pt, c in sorted(peaks, key=lambda x: -x[1]):
        print(f"  {l:<16} 跳高 {h:+.3f} m   峰值 {pk:.3f} m @ t={pt:+.2f}s")
    if settle_peaks:
        print("\n触发前站立波动峰值 (落地缓冲, 非跳跃):")
        for l, h in settle_peaks:
            if h > 0.05:
                print(f"  {l:<16} +{h:.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
