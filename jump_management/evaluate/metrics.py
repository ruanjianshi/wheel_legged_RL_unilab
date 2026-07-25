"""跳跃评估专用指标计算."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SummaryMetrics:
    """汇总指标."""

    model_name: str
    num_episodes: int
    success_rate: float = 0.0
    survival_rate: float = 0.0
    avg_jump_distance: float = 0.0
    avg_jump_height: float = 0.0
    avg_wheel_slip: float = 0.0
    avg_jumps_per_ep: float = 0.0
    # 各场景明细
    per_scenario: dict[str, dict[str, float]] = None  # type: ignore

    def __post_init__(self):
        if self.per_scenario is None:
            self.per_scenario = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_episodes": self.num_episodes,
            "success_rate": self.success_rate,
            "survival_rate": self.survival_rate,
            "avg_jump_distance": self.avg_jump_distance,
            "avg_jump_height": self.avg_jump_height,
            "avg_wheel_slip": self.avg_wheel_slip,
            "avg_jumps_per_ep": self.avg_jumps_per_ep,
            "per_scenario": self.per_scenario,
        }


def aggregate_scenario(results: list[dict]) -> dict[str, float]:
    """聚合单个场景的多回合结果."""
    n = len(results)
    if n == 0:
        return {}

    successes = [r["success"] for r in results]
    survived = [r["survived"] for r in results]
    distances = [r["avg_jump_distance"] for r in results if r["avg_jump_distance"] > 0]
    heights = [r["avg_jump_height"] for r in results if r["avg_jump_height"] > 0]
    slips = [r["avg_wheel_slip"] for r in results if r["avg_wheel_slip"] < float("inf")]
    jumps = [r["num_jumps"] for r in results]

    return {
        "num_episodes": n,
        "success_rate": float(np.mean(successes)),
        "survival_rate": float(np.mean(survived)),
        "avg_jump_distance": float(np.mean(distances)) if distances else 0.0,
        "avg_jump_height": float(np.mean(heights)) if heights else 0.0,
        "avg_wheel_slip": float(np.mean(slips)) if slips else float("inf"),
        "avg_jumps_per_ep": float(np.mean(jumps)),
    }


def compute_summary(
    model_name: str,
    all_results: dict[str, list[dict]],
) -> SummaryMetrics:
    """计算汇总指标."""
    all_eps = [r for results in all_results.values() for r in results]
    summary = SummaryMetrics(model_name=model_name, num_episodes=len(all_eps))

    summary.per_scenario = {}
    for scenario, results in all_results.items():
        summary.per_scenario[scenario] = aggregate_scenario(results)

    # 全局汇总
    scores = {
        "success": [],
        "survival": [],
        "distance": [],
        "height": [],
        "slip": [],
        "jumps": [],
    }
    for s in summary.per_scenario.values():
        scores["success"].append(s["success_rate"])
        scores["survival"].append(s["survival_rate"])
        if s["avg_jump_distance"] > 0:
            scores["distance"].append(s["avg_jump_distance"])
        if s["avg_jump_height"] > 0:
            scores["height"].append(s["avg_jump_height"])
        if s["avg_wheel_slip"] < float("inf"):
            scores["slip"].append(s["avg_wheel_slip"])
        scores["jumps"].append(s["avg_jumps_per_ep"])

    summary.success_rate = float(np.mean(scores["success"])) if scores["success"] else 0
    summary.survival_rate = float(np.mean(scores["survival"])) if scores["survival"] else 0
    summary.avg_jump_distance = float(np.mean(scores["distance"])) if scores["distance"] else 0
    summary.avg_jump_height = float(np.mean(scores["height"])) if scores["height"] else 0
    summary.avg_wheel_slip = float(np.mean(scores["slip"])) if scores["slip"] else float("inf")
    summary.avg_jumps_per_ep = float(np.mean(scores["jumps"])) if scores["jumps"] else 0

    return summary


def save_summary(summary: SummaryMetrics, path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)


def load_summary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
