"""Automated analysis report generator.

Produces formatted Markdown reports from evaluation JSON results.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_flat_walk_report(
    result: dict[str, Any],
    output_path: str | Path,
):
    """Generate a structured analysis report for flat walk evaluation."""
    results = result.get("results", {})
    run = result.get("run", "")
    ckpt = result.get("checkpoint", 0)

    def m(scenario: str, metric: str, default=0.0):
        return results.get(scenario, {}).get("metrics", {}).get(metric, default)

    lines = [
        "# XqRobotV2 平坦地面行走 — PPO 策略评估报告",
        "",
        f"**模型**: {run} @ iter={ckpt}",
        f"**评估时间**: {result.get('evaluated_at', 'N/A')}",
        f"**耗时**: {result.get('elapsed_sec', 0):.0f}s",
        "",
        "---",
        "",
        "## 1. 前向速度跟踪 (Vx Tracking)",
        "",
        "| 指令 Vx | 实际 Vx | RMSE | 跟踪比 | Vy串扰 | 评估 |",
        "|---------|---------|------|--------|--------|------|",
    ]

    for vx_cmd in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]:
        key = f"vx={vx_cmd}"
        avg = m(key, "avg_vx")
        rmse = m(key, "vx_tracking_rmse")
        ratio = m(key, "vel_tracking_ratio")
        coupling = m(key, "vel_coupling")
        cmd_val = float(vx_cmd)

        if ratio > 1.3:
            tag = "🔶 超调"
        elif 0.85 <= ratio <= 1.15 and rmse < 0.1:
            tag = "✅ 优秀"
        elif 0.6 <= ratio < 0.85:
            tag = "⚠️ 不足"
        elif rmse > 0.2:
            tag = "❌ 不足"
        else:
            tag = "🔶"

        lines.append(
            f"| {cmd_val:.1f} m/s | {avg:.3f} | {rmse:.3f} | {ratio:.2f}x | {coupling:.3f} | {tag} |"
        )

    lines += [
        "",
        "## 2. Vx/Vy 解耦",
        "",
        "| 指令 | Vy串扰 | 评估 |",
        "|------|--------|------|",
    ]
    for vx_cmd in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]:
        c = m(f"vx={vx_cmd}", "vel_coupling")
        tag = "✅" if c < 0.05 else ("⚠️" if c < 0.1 else "❌")
        lines.append(f"| Vx={vx_cmd} | {c:.3f} | {tag} |")

    lines += [
        "",
        "## 3. 横向移动 (Vy Tracking)",
        "",
        "| 指令 Vy | 实际 Vy | 实际 Vx | RMSE | 评估 |",
        "|---------|---------|---------|------|------|",
    ]
    for vy_cmd in ["0.1", "0.2", "0.3"]:
        key = f"vy={vy_cmd}"
        avg_vy = m(key, "avg_vy")
        avg_vx = m(key, "avg_vx")
        rmse = m(key, "vy_tracking_rmse")
        tag = "✅" if rmse < 0.1 else ("⚠️" if rmse < 0.2 else "❌")
        lines.append(f"| {vy_cmd} m/s | {avg_vy:.3f} | {avg_vx:.3f} | {rmse:.3f} | {tag} |")

    lines += [
        "",
        "## 4. 后退行走",
        "",
        "| 指令 Vx | 实际 Vx | RMSE | 跟踪比 | 评估 |",
        "|---------|---------|------|--------|------|",
    ]
    for vx_cmd in ["-0.3", "-0.6"]:
        key = f"vx={vx_cmd}"
        avg = m(key, "avg_vx")
        rmse = m(key, "vx_tracking_rmse")
        ratio = m(key, "vel_tracking_ratio")
        tag = "✅" if rmse < 0.15 else ("⚠️" if rmse < 0.3 else "❌")
        lines.append(f"| {vx_cmd} m/s | {avg:.3f} | {rmse:.3f} | {ratio:.2f}x | {tag} |")

    lines += [
        "",
        "## 5. 偏航控制",
        "",
        "| 指令 Vyaw | RMSE | 评估 |",
        "|-----------|------|------|",
    ]
    for vyaw in ["0.5", "1.0"]:
        key = f"vyaw={vyaw}"
        rmse = m(key, "vyaw_tracking_rmse")
        tag = "✅" if rmse < 0.05 else ("⚠️" if rmse < 0.1 else "❌")
        lines.append(f"| {vyaw} rad/s | {rmse:.3f} | {tag} |")

    lines += [
        "",
        "## 6. 组合指令",
        "",
        "| 组合 | Vx RMSE | Vy RMSE | Vyaw RMSE | 评估 |",
        "|------|---------|---------|-----------|------|",
    ]
    for combo_key, combo_label in [
        ("vx=0.3_vy=0.2", "Vx=0.3+Vy=0.2"),
        ("vx=0.3_vyaw=0.5", "Vx=0.3+Vyaw=0.5"),
        ("vx=0.3_vy=0.2_vyaw=0.5", "全组合"),
    ]:
        vx_r = m(combo_key, "vx_tracking_rmse")
        vy_r = m(combo_key, "vy_tracking_rmse")
        vyaw_r = m(combo_key, "vyaw_tracking_rmse")
        avg = (vx_r + vy_r + vyaw_r) / 3 if all(v > 0 for v in [vx_r, vy_r, vyaw_r]) else 1.0
        tag = "✅" if avg < 0.10 else ("⚠️" if avg < 0.20 else "❌")
        lines.append(f"| {combo_label} | {vx_r:.3f} | {vy_r:.3f} | {vyaw_r:.3f} | {tag} |")

    # Stability summary
    heights = [m(f"vx={v}", "base_height_mean") for v in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]]
    h_std = [m(f"vx={v}", "base_height_std") for v in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]]
    orients = [
        m(f"vx={v}", "orientation_rmse_deg") for v in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    ]

    lines += [
        "",
        "## 7. 稳定性汇总",
        "",
        "| 指标 | 均值 | 范围 | 评估 |",
        "|------|------|------|------|",
        f"| 基座高度 | {sum(heights) / len(heights):.3f} m | {min(heights):.3f}-{max(heights):.3f} | "
        f"{'⚠️ 偏低' if sum(heights) / len(heights) < 0.60 else '✅'} |",
        f"| 高度波动 | {sum(h_std) / len(h_std):.4f} m | {min(h_std):.4f}-{max(h_std):.4f} | "
        f"{'✅' if sum(h_std) / len(h_std) < 0.03 else '⚠️'} |",
        f"| 姿态 RMSE | {sum(orients) / len(orients):.1f}° | {min(orients):.1f}-{max(orients):.1f} | "
        f"{'✅' if sum(orients) / len(orients) < 5 else '⚠️'} |",
        "",
        "## 8. 运动质量",
        "",
        "| 指令 | 动作平滑度 | 关节速度 | 步态对称 |",
        "|------|-----------|---------|---------|",
    ]
    for vx_cmd in ["0.1", "0.3", "0.6"]:
        key = f"vx={vx_cmd}"
        smooth = m(key, "action_smoothness")
        jvel = m(key, "joint_velocity_mean")
        sym = m(key, "gait_symmetry")
        lines.append(f"| Vx={vx_cmd} | {smooth:.3f} | {jvel:.3f} rad/s | {sym:.3f} |")

    # Overall score
    lines += [
        "",
        "---",
        f"*报告由 assess 框架自动生成 — {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


def generate_report(
    result: dict[str, Any] | str | Path,
    output_path: str | Path,
    task: str = "flat_walk",
) -> Path:
    """Auto-generate analysis report from evaluation result."""
    if isinstance(result, (str, Path)):
        with open(result) as f:
            result = json.load(f)

    if task == "flat_walk":
        return generate_flat_walk_report(result, output_path)
    else:
        lines = [f"# {task} Evaluation Report", "", f"**Result**: {Path(str(output_path)).name}"]
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return Path(output_path)
