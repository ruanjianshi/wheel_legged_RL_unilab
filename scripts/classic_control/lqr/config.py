"""LQR 任务轨配置 — conf/lqr/ (完全自包含, 不干涉 MPC).

含: robot.yaml + commands.yaml + task/ (env 映射) + config.yaml (LQR 权重)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.classic_control.common.config import ROOT, commands_params

# ★ LQR 轨完整配置目录
CONF_DIR = ROOT / "conf" / "lqr"
_LQR_CONF = CONF_DIR / "config.yaml"


@dataclass
class LqrConfig:
    """LQR 矢状面权重 (★ 直接作增益) + 偏航/腿控字段 (控制器读取)."""

    q_theta: float = 100.0  # θ 增益 (★ 修正方向后实测最优)
    q_theta_dot: float = 20.0  # θ̇ 增益
    q_v: float = 80.0  # v 增益 (漂移控制关键)
    q_x: float = 30.0  # 位置增益
    q_z: float = 8.0  # LQI 速度积分增益
    r: float = 1.0  # 控制权重 (保留, LQR 未用)

    # 偏航 (共享命令表, 控制器经 cfg 读取)
    track_width: float = 0.38
    k_yaw: float = 3.0
    wheel_vel_max: float = 25.0
    cmd_ramp_s: float = 1.5

    # 腿控 (P3 膝高度伺服)
    height_kp: float = 0.8
    height_kd: float = 0.3
    height_ki: float = 0.2
    height_smoothing: float = 0.97
    leg_balance_kp: float = -0.25
    l0_margin: float = 0.03


_FIELDS = frozenset(LqrConfig.__dataclass_fields__)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_params(task_key: str = "walk_flat") -> dict:
    """conf/lqr LQR 权重 + 本轨命令参数 → 扁平 dict."""
    alg = _load_yaml(_LQR_CONF)["sagittal"]
    return {**alg, **commands_params(task_key, conf_dir=CONF_DIR)}


def build_config(
    task_key: str = "walk_flat", overrides: dict | None = None
) -> tuple[LqrConfig, dict]:
    """合并默认 + CLI 覆盖 → (LqrConfig, merged). merged 保留 smoothing/sign 等."""
    p = default_params(task_key)
    if overrides:
        p = {**p, **overrides}
    cfg = LqrConfig(**{k: v for k, v in p.items() if k in _FIELDS})
    return cfg, p
