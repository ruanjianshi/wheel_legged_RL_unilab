"""MPC 任务轨配置 — conf/mpc/ (完全自包含, 不干涉 LQR).

含: robot.yaml + commands.yaml + task/ (env 映射) + config.yaml (MPC 权重)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts.classic_control.common.config import ROOT, commands_params

# ★ MPC 轨完整配置目录
CONF_DIR = ROOT / "conf" / "mpc"
_MPC_CONF = CONF_DIR / "config.yaml"


@dataclass
class MpcConfig:
    """MPC 代价权重 + 模型/时域/约束/积分参数 + 偏航/腿控字段."""

    # 代价权重 (sagittal 作 LQR 代价, 非增益)
    q_theta: float = 100.0
    q_theta_dot: float = 20.0
    q_v: float = 80.0
    q_x: float = 30.0
    q_z: float = 8.0  # P2 积分状态权重
    r: float = 1.0  # 控制代价

    # MPC 模型/时域/约束
    mpc_horizon: int = 20
    u_max: float = 30.0  # 轮速命令限 (rad/s, 匹配 wheel_vel_max clip)
    theta_max: float = 0.35  # 倾角限 (rad)
    v_max: float = 2.5  # 轮速限 (m/s)
    wheel_vel_max: float = 25.0
    tau: float = 0.059  # 轮速度伺服时间常数 (s), 实测
    alpha: float = 24.4  # 倒立摆 α (θ̈=αθ), 自由落体实测
    beta: float = -2.5  # 轮加速耦合 β (θ̈=β·v̇_wheel)
    terminal_lqr: bool = True  # 末端 LQR 代价
    r_rate: float = 0.5  # 控制变化率惩罚
    integral_gain: float = 1.5  # P2 积分参考偏置增益
    model_file: str = "logs/classic/mpc_plant_bb.npz"  # 黑箱模型 (P2+ 覆盖解析)

    # 偏航/命令/腿控 (共享命令表, 控制器经 cfg 读取)
    track_width: float = 0.38
    k_yaw: float = 3.0
    cmd_ramp_s: float = 1.5
    height_kp: float = 0.8
    height_kd: float = 0.3
    height_ki: float = 0.2
    height_smoothing: float = 0.97
    leg_balance_kp: float = -0.25
    l0_margin: float = 0.03


_FIELDS = frozenset(MpcConfig.__dataclass_fields__)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_params(task_key: str = "walk_flat") -> dict:
    """conf/mpc MPC 权重 + 本轨命令参数 → 扁平 dict."""
    c = _load_yaml(_MPC_CONF)
    return {**c["sagittal"], **c["mpc"], **commands_params(task_key, conf_dir=CONF_DIR)}


def build_config(
    task_key: str = "walk_flat", overrides: dict | None = None
) -> tuple[MpcConfig, dict]:
    """合并默认 + CLI 覆盖 → (MpcConfig, merged). merged 保留 smoothing/sign 等."""
    p = default_params(task_key)
    if overrides:
        p = {**p, **overrides}
    cfg = MpcConfig(**{k: v for k, v in p.items() if k in _FIELDS})
    return cfg, p
