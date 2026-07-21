#!/usr/bin/env python3
"""表3: 各子任务分解性能 (Wheeled-SRL完成训练后评估).

生成 LaTeX 格式的表格.
"""
from __future__ import annotations

import json


def generate_task_breakdown(
    results_base: str = "results/srl_full",
    out_path: str = "paper/table_task_breakdown.tex",
):
    """生成任务分解表."""
    scenarios = {
        "fix_01m": {"label": "固定0.1m", "target": 0.1},
        "fix_02m": {"label": "固定0.2m", "target": 0.2},
        "fix_03m": {"label": "固定0.3m", "target": 0.3},
        "random": {"label": "随机$[0, 0.3]$m", "target": None},
        "platform": {"label": "跳台0.15m", "target": 0.15},
    }

    rows = []
    for sc_name, sc_info in scenarios.items():
        try:
            with open(f"{results_base}/{sc_name}.json") as f:
                data = json.load(f)

            success = sum(1 for e in data if e["success"]) / len(data) if data else 0
            slips = [e.get("avg_wheel_slip", 0) for e in data if e["avg_wheel_slip"] < float("inf")]
            avg_slip = sum(slips) / len(slips) if slips else 0
            avg_dist = sum(e["avg_jump_distance"] for e in data) / len(data) if data else 0

            rows.append((sc_info["label"], success, avg_slip, avg_dist))
        except FileNotFoundError:
            rows.append((sc_info["label"], 0.0, 0.0, 0.0))

    latex = r"""\begin{table}[htbp]
    \centering
    \caption{各子任务分解性能（Wheeled-SRL完成训练后评估）}
    \label{tab:task_breakdown}
    \begin{tabular}{lccc}
        \toprule
        子任务 & 成功率 & 着陆轮地误差(m/s) & 平均跳距(m) \\
        \midrule
"""
    for label, sr, slip, dist in rows:
        latex += f"        {label} & {sr:.3f} & {slip:.2f} & {dist:.3f} \\\\\n"

    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""

    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Saved: {out_path}")
    print(latex)


if __name__ == "__main__":
    generate_task_breakdown()
