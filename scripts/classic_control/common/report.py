"""指标报告: stdout 表格 + markdown (经典控制轨)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.classic_control.common.metrics import threshold_for


def print_metrics(header: str, metrics: dict[str, float], phase: int) -> None:
    print(f"\n=== {header} ===")
    thr = threshold_for(phase)
    for key, val in sorted(metrics.items()):
        line = f"  {key:<18} {val:8.3f}"
        if key in thr:
            limit, direction, unit = thr[key]
            ok = val < limit if direction == "<" else val > limit
            mark = "✅" if ok else "❌"
            line += f"   阈值 {direction} {limit}{unit} {mark}"
        print(line)


def write_report(
    out_path: str | Path,
    title: str,
    metrics: dict[str, float],
    phase: int,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    thr = threshold_for(phase)
    lines = [f"# {title}", "", "| 指标 | 值 | 阈值 | 判定 |", "|---|---|---|---|"]
    for key, val in sorted(metrics.items()):
        if key in thr:
            limit, direction, unit = thr[key]
            ok = val < limit if direction == "<" else val > limit
            mark = "✅" if ok else "❌"
            lines.append(f"| {key} | {val:.3f} | {direction}{limit}{unit} | {mark} |")
        else:
            lines.append(f"| {key} | {val:.3f} | — | — |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  → 报告: {out_path}")
