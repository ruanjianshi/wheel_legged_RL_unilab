"""MPC×SAC 融合控制分支配置 — conf/fusion_control/mpc_sac/ (完全自包含).

含: robot.yaml + commands.yaml + task/ (env 映射) + config.yaml
(低层 mpc + 高层 sac + cmd_scale + desired + reward + train)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scripts.classic_control.common.config import ROOT, commands_params

# ★ 融合控制分支完整配置目录 (自包含)
CONF_DIR = ROOT / "conf" / "fusion_control" / "mpc_sac"
_CFG_YAML = CONF_DIR / "config.yaml"


@dataclass
class MpcSacConfig:
    """融合控制配置: 低层 MPC + 高层 SAC + 命令缩放 + 奖励 + 训练."""

    # ── 低层 MPC (分支自带, 冻结) ──
    mpc_horizon: int = 20
    u_max: float = 30.0
    theta_max: float = 0.35
    v_max: float = 2.5
    wheel_vel_max: float = 25.0
    tau: float = 0.059
    alpha: float = 24.4
    beta: float = -2.5
    terminal_lqr: bool = True
    r_rate: float = 0.5
    integral_gain: float = 1.5
    model_file: str = "logs/classic/mpc_plant_bb.npz"
    q_theta: float = 100.0
    q_theta_dot: float = 20.0
    q_v: float = 80.0
    q_x: float = 30.0
    q_z: float = 8.0
    r: float = 1.0

    # ── 高层 SAC ──
    obs_dim_flat: int = 11
    obs_dim_rough: int = 9
    action_dim_flat: int = 3
    action_dim_rough: int = 2
    actor_hidden_dims: tuple = (256, 128)
    critic_hidden_dims: tuple = (256, 128)
    activation: str = "relu"
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    alpha_init: float = 0.2
    target_entropy_ratio: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    init_noise_std: float = 0.4
    min_action: float = -1.0
    max_action: float = 1.0
    batch_size: int = 256
    replay_buffer_capacity: int = 200000
    updates_per_step: int = 2
    obs_normalization: bool = True

    # ── 残差命令 (cmd = des + res_scale·a) ──
    res_vx: float = 0.2
    res_vyaw: float = 0.05
    res_height: float = 0.02
    res_vx_rough: float = 0.3
    res_vyaw_rough: float = 0.1
    res_tsk: float = 0.0

    # ── 期望命令采样 ──
    resample_s_flat: float = 3.0
    resample_s_rough: float = 6.0
    vx_range_flat: tuple = (-0.6, 0.6)
    vyaw_range_flat: tuple = (-0.05, 0.05)
    height_range_flat: tuple = (0.49, 0.545)
    vx_range_rough: tuple = (-0.6, 0.6)
    vyaw_range_rough: tuple = (-0.1, 0.1)

    # ── 奖励 ──
    w_alive: float = 0.8
    w_vx: float = 3.0
    sigma_vx: float = 0.25
    w_vyaw: float = 1.0
    sigma_vyaw: float = 0.4
    w_h: float = 1.5
    sigma_h: float = 0.02
    w_theta: float = -2.0
    w_omega: float = -0.5
    w_corr: float = -0.3
    w_energy: float = -0.001
    w_cmd_rate: float = -0.1

    # ── 训练 ──
    num_envs: int = 64
    max_episode_steps: int = 1000
    max_iterations: int = 3000
    eval_interval: int = 200
    save_interval: int = 500
    log_dir: str = "logs/fusion_control/mpc_sac"
    seed: int = 42
    device: str = "auto"

    # ── 偏航/命令/腿控 (经 commands_params 合并, 供低层 MPC) ──
    track_width: float = 0.38
    k_yaw: float = 3.0
    cmd_ramp_s: float = 1.5
    height_kp: float = 0.8
    height_kd: float = 0.3
    height_ki: float = 0.2
    height_smoothing: float = 0.97
    leg_balance_kp: float = -0.25
    l0_margin: float = 0.03
    smoothing: float = 0.85
    sign: float = 1.0

    # ── 供低层 MPC 的额外字段 ──
    phase_flat: int = 3
    phase_rough: int = 4
    # ★ 粗糙地形用缓坡版 (默认 rough 地形过难, MPC 基线站不住)
    rough_gentle: bool = True

    # ── 高层 obs 长度 (组合时用) ──
    use_height_scan: bool = False


_FIELDS = frozenset(MpcSacConfig.__dataclass_fields__)


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def default_params(task_key: str = "walk_flat") -> dict:
    """conf/fusion_control/mpc_sac 全部段 → 扁平 dict (含命令参数)."""
    c = _load_yaml(_CFG_YAML)
    p: dict = {}
    for sec in ("mpc", "sac", "cmd_scale", "desired", "reward", "train"):
        p.update(c.get(sec, {}))
    p.update(commands_params(task_key, conf_dir=CONF_DIR))
    return p


def build_config(
    task_key: str = "walk_flat", overrides: dict | None = None
) -> tuple[MpcSacConfig, dict]:
    """合并默认 + 覆盖 → (cfg, merged). merged 保留 smoothing/sign 等, 供低层 MPC."""
    p = default_params(task_key)
    if overrides:
        p = {**p, **overrides}
    cfg = MpcSacConfig(**{k: v for k, v in p.items() if k in _FIELDS})
    return cfg, p
