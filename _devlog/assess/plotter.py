"""Visualization module for policy evaluation results.

Generates publication-quality plots for:
- Velocity tracking curves (commanded vs actual)
- Multi-metric radar/spider charts for model comparison
- Bar charts across scenarios and metrics
- Stability timeline plots
- Gait phase portraits
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Use non-interactive backend
import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402

# ── Style ──
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

COLORS = {
    "red": "#E74C3C",
    "blue": "#3498DB",
    "green": "#2ECC71",
    "orange": "#F39C12",
    "purple": "#9B59B6",
    "teal": "#1ABC9C",
    "gray": "#95A5A6",
    "dark": "#2C3E50",
}


def plot_velocity_tracking(
    records: dict[str, Any],
    output_path: str | Path,
    title: str = "Velocity Tracking",
):
    """Time-series plot of commanded vs actual velocity per scenario."""
    n = len(records)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (name, rec) in enumerate(records.items()):
        ax = axes[idx // cols, idx % cols]
        t = np.array(rec.timestamps)
        cmd_x, cmd_y = rec.cmd[0], rec.cmd[1]
        ax.plot(t, rec.vx, color=COLORS["red"], label=f"Vx (cmd={cmd_x:.1f})", alpha=0.9)
        ax.plot(t, rec.vy, color=COLORS["blue"], label=f"Vy (cmd={cmd_y:.1f})", alpha=0.9)
        ax.axhline(y=cmd_x, color=COLORS["red"], linestyle="--", alpha=0.4, lw=1)
        ax.axhline(y=cmd_y, color=COLORS["blue"], linestyle="--", alpha=0.4, lw=1)
        ax.set_title(name[:30])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_radar(
    models_data: dict[str, dict[str, float]],
    output_path: str | Path,
    title: str = "Policy Comparison",
    metrics_subset: list[str] | None = None,
):
    """Spider/radar chart comparing multiple models across metrics."""
    if not models_data:
        return

    # Select metrics present in all models
    all_metrics = set.intersection(*[set(d.keys()) for d in models_data.values()])
    if metrics_subset:
        all_metrics = all_metrics & set(metrics_subset)
    metrics = sorted(all_metrics)
    if len(metrics) < 2:
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_list = list(COLORS.values())

    for i, (model_name, data) in enumerate(models_data.items()):
        values = [data.get(m, 0) or 0 for m in metrics]
        values += values[:1]
        color = colors_list[i % len(colors_list)]
        ax.fill(angles, values, alpha=0.15, color=color)
        ax.plot(angles, values, "o-", color=color, lw=2, label=model_name[:25], markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_title(title, fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_bars(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: str | Path,
    title: str = "Metrics per Scenario",
):
    """Grouped bar chart: one subplot per metric, bars = scenarios."""
    # Collect all metrics across all scenarios
    all_scenarios = list(results.keys())
    if not all_scenarios:
        return

    # Get all metric names (first scenario)
    first_data = results[all_scenarios[0]]
    if isinstance(first_data, dict):
        metric_names = list(first_data.get("metrics", {}).keys())
    else:
        return

    if not metric_names:
        return

    n_metrics = len(metric_names)
    cols = min(n_metrics, 4)
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes)

    for i, metric in enumerate(metric_names):
        ax = (
            axes[i // cols]
            if cols == 1
            else axes[i // cols][i % cols]
            if rows > 1
            else axes[i % cols]
        )
        values = []
        labels = []
        for s in all_scenarios:
            v = results[s].get("metrics", {}).get(metric)
            if v is not None:
                values.append(v)
                labels.append(s[:15])
        bars = ax.bar(range(len(values)), values, color=COLORS["blue"], alpha=0.7, width=0.6)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    for idx in range(n_metrics, rows * cols):
        axes.flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_stability_timeline(
    records: dict[str, Any],
    output_path: str | Path,
    title: str = "Stability Timeline",
):
    """Plot base height and orientation over time for all scenarios."""
    n = len(records)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.2 * n))
    if n == 1:
        axes = np.array([axes])

    for idx, (name, rec) in enumerate(records.items()):
        t = np.array(rec.timestamps)
        # Base height
        axes[idx, 0].plot(t, rec.base_z, color=COLORS["dark"], alpha=0.8)
        axes[idx, 0].axhline(y=0.65, color=COLORS["gray"], linestyle="--", alpha=0.5)
        axes[idx, 0].set_ylabel("Height (m)")
        axes[idx, 0].set_title(f"{name[:25]} — Height")
        axes[idx, 0].grid(True, alpha=0.3)
        # Orientation
        axes[idx, 1].plot(
            t, np.rad2deg(rec.base_roll), color=COLORS["red"], label="Roll", alpha=0.7
        )
        axes[idx, 1].plot(
            t, np.rad2deg(rec.base_pitch), color=COLORS["blue"], label="Pitch", alpha=0.7
        )
        axes[idx, 1].set_ylabel("Angle (deg)")
        axes[idx, 1].set_title(f"{name[:25]} — Orientation")
        axes[idx, 1].legend(fontsize=7)
        axes[idx, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_metric_comparison(
    comparisons: list[dict[str, Any]],
    output_path: str | Path,
    metric_key: str = "vx_tracking_rmse",
    title: str = "Cross-Model Comparison",
):
    """Compare a single metric across multiple models, scenarios as groups."""
    if not comparisons:
        return

    # Assemble data: {scenario: {model_label: value}}
    scenario_metrics: dict[str, dict[str, float]] = {}
    for comp in comparisons:
        label = comp.get("label", "unknown")
        for sname, sdata in comp.get("results", {}).items():
            if sname not in scenario_metrics:
                scenario_metrics[sname] = {}
            scenario_metrics[sname][label] = sdata.get("metrics", {}).get(metric_key, 0)

    scenarios = list(scenario_metrics.keys())
    models = sorted(set().union(*[d.keys() for d in scenario_metrics.values()]))
    colors_list = list(COLORS.values())

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.5), 5))
    x = np.arange(len(scenarios))
    bar_w = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        vals = [scenario_metrics[s].get(model, 0) for s in scenarios]
        ax.bar(
            x + i * bar_w,
            vals,
            bar_w,
            color=colors_list[i % len(colors_list)],
            alpha=0.8,
            label=model[:20],
        )

    ax.set_xticks(x + bar_w * (len(models) - 1) / 2)
    ax.set_xticklabels([s[:15] for s in scenarios], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_gait_phase(
    records: dict[str, Any],
    output_path: str | Path,
    title: str = "Gait Phase Portrait",
):
    """Plot joint angle phase portraits (angle vs angular velocity)."""
    # Pick one scenario for gait analysis (first with leg data)
    target = None
    for name, rec in records.items():
        if rec.leg_pos and rec.leg_vel:
            target = rec
            break
    if target is None:
        return

    pos_arr = np.array([p for p in target.leg_pos])
    vel_arr = np.array([v for v in target.leg_vel])
    if len(pos_arr) < 2:
        return

    joint_names = ["L_hip", "L_thigh", "L_calf", "R_hip", "R_thigh", "R_calf"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i < pos_arr.shape[1]:
            ax.plot(pos_arr[:, i], vel_arr[:, i], color=COLORS["blue"], alpha=0.4, lw=0.5)
            ax.scatter(
                pos_arr[0, i], vel_arr[0, i], color=COLORS["green"], s=30, zorder=5, label="start"
            )
            ax.scatter(
                pos_arr[-1, i], vel_arr[-1, i], color=COLORS["red"], s=30, zorder=5, label="end"
            )
            ax.set_xlabel("Position (rad)")
            ax.set_ylabel("Velocity (rad/s)")
            ax.set_title(joint_names[i])
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title} — {target.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
