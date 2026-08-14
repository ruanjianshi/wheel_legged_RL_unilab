"""MPC×SAC 融合分支 — 指标: 复用 common.metrics + 融合专用 (约束违反率/介入率)."""

from __future__ import annotations

import numpy as np

from scripts.classic_control.common import metrics as common_metrics
from scripts.fusion_control.mpc_sac.config import MpcSacConfig


def compute(record: list[dict], phase: int, cfg: MpcSacConfig) -> dict:
    """标准指标 (common) + 融合专用指标."""
    m = common_metrics.compute(record, phase)
    m.update(compute_fusion_metrics(record, cfg))
    return m


def compute_fusion_metrics(record: list[dict], cfg: MpcSacConfig) -> dict:
    """融合专用: 安全约束违反率 / 高层发令 vs 期望 / 低层求解耗时."""
    theta = np.asarray([r["sensors"]["theta"] for r in record], dtype=np.float64)
    v = np.asarray([r["sensors"]["v"] for r in record], dtype=np.float64)
    cmd_hi_vx = np.asarray([r["cmd_hi"][0] for r in record], dtype=np.float64)
    des_vx = np.asarray([r["cmd"][0] for r in record], dtype=np.float64)
    solve_ms = float(np.mean([r.get("solve_ms", 0.0) for r in record])) if record else 0.0
    return {
        "theta_max_viol": float(np.mean(np.abs(theta) > cfg.theta_max)),
        "v_max_viol": float(np.mean(np.abs(v) > cfg.v_max)),
        "cmd_track_err": float(np.mean(np.abs(cmd_hi_vx - des_vx))),
        "solve_ms_mean": float(solve_ms),
    }


def threshold_for(phase: int) -> dict:
    return common_metrics.threshold_for(phase)
