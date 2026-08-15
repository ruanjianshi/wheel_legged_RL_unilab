"""MPC×SAC 融合分支 — env 构建与传感器 (自包含: 读本分支 conf/task).

移植经典轨 scripts/classic_control/common/rollout.py 的 build_env/read_sensors,
增加 conf_dir 参数 (默认指向本分支 conf) + 向量化 read_sensors_batch (训练用)。
StandingProvider/CmdSchedule 复用 common (共享只读)。
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import yaml

from scripts.classic_control.common.config import (
    HIP_Z_OFFSET,
    STANDING_ANGLES,
    STANDING_BASE_Z,
    VMC,
    WHEEL_R,
    env_conf,
)
from scripts.classic_control.common.rollout import CmdSchedule, StandingProvider
from scripts.fusion_control.mpc_sac.config import CONF_DIR
from unilab.base import registry
from unilab.base.registry import ensure_registries

ROOT = CONF_DIR.parents[2]

_L1 = float(VMC["l1"])
_L2 = float(VMC["l2"])
_HIP_SIGN = np.asarray(VMC["hip_sign"], dtype=np.float64)
_KNEE_SIGN = np.asarray(VMC["knee_sign"], dtype=np.float64)
_C1 = float(VMC["c1"])
_C2 = float(VMC["c2"])
_OFF = float(VMC["offset"])


def _fk_legs_batch(dof_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """向量化腿 FK: dof_pos (N,8) → (L0 (N,2), theta0 (N,2))."""
    qp = dof_pos[:, [1, 4]]  # L_pitch, R_pitch
    qk = dof_pos[:, [2, 5]]  # L_knee, R_knee
    t1 = _HIP_SIGN[None, :] * qp + _C1
    t2 = _KNEE_SIGN[None, :] * qk + _C2
    ex = _OFF + _L1 * np.cos(t1) + _L2 * np.cos(t1 + t2)
    ey = _L1 * np.sin(t1) + _L2 * np.sin(t1 + t2)
    L0 = np.sqrt(ex * ex + ey * ey)
    theta0 = np.arctan2(ey, ex) - np.pi / 2.0
    return L0, theta0


def _env_conf(task_key: str) -> dict:
    """本分支 task 配置的 env 段 (自包含)."""
    return env_conf(task_key, conf_dir=CONF_DIR)


def _load_conf(task_key: str) -> dict:
    with open(ROOT / _env_conf(task_key)["mujoco_yaml"], encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_env(
    task_key: str = "walk_flat",
    num_envs: int = 1,
    cmd: np.ndarray | None = None,
    lock_hip_roll: bool = True,
    rough_gentle: bool = False,
) -> Any:
    """构建确定性环境 (关课程/域随机化/命令重采样/平滑/噪声). 配置来自本分支 conf.

    rough_gentle=True: ★ 粗糙地形用缓坡版 (无陡坡/大波), 让 MPC 基线能站稳 — 默认 rough 地形
    (波 18cm + 坡 0.5) 对 MPC 基线过难 (经典 P4 0%), 融合训练需要可站的地形。
    """
    ensure_registries()
    cfg = _load_conf(task_key)
    override: dict[str, Any] = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))
    override["curriculum"] = {"enabled": False}
    override["max_episode_seconds"] = 1000.0
    override["commands"] = {**override.get("commands", {}), "resampling_time": 0.0}
    override["control_config"] = {**override.get("control_config", {}), "action_smoothing": 0.0}
    override["noise_config"] = {"level": 0.0}
    override["domain_rand"] = {
        **override.get("domain_rand", {}),
        "randomize_init_yaw": False,
        "randomize_base_mass": False,
        "randomize_ground_friction": False,
        "randomize_kp": False,
        "randomize_kd": False,
        "random_com": False,
        "randomize_leg_length": False,
        "push_robots": False,
    }
    if task_key == "walk_rough":
        override["terrain_curriculum"] = {"enabled": False}
        override["terrain_scan"] = {"enabled": False}
        if rough_gentle:
            # ★ 缓坡地形: 平地 + 小 bump + 缓波, 无陡坡 (默认有 hf_pyramid_slope 坡 0.5)
            #   地形生成器在 scene.terrain.generator (XqRobotWLRoughTerrainCfg.sub_terrains)
            from unilab.terrains import flat, random_rough, wave_terrain

            override["scene"] = {
                "terrain": {
                    "generator": {
                        "sub_terrains": {
                            "flat": flat(proportion=0.3),
                            "random_rough": random_rough(
                                proportion=0.4, noise_range=(0.01, 0.04), noise_step=0.01,
                                border_width=0.2,
                            ),
                            "wave_terrain": wave_terrain(
                                proportion=0.3, amplitude_range=(0.02, 0.08), num_waves=3,
                                border_width=0.2,
                            ),
                        }
                    }
                }
            }

    env = registry.make(
        _env_conf(task_key)["task_name"],
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )
    cmd_dim = int(_env_conf(task_key)["command_dim"])
    if cmd is None:
        cmd = np.zeros(cmd_dim, dtype=np.float64)
        if cmd_dim == 5:
            cmd[4] = STANDING_BASE_Z
    env._dr_manager._provider = StandingProvider(cmd[:cmd_dim])  # type: ignore[union-attr]
    env.set_autoreset(False)
    if lock_hip_roll:
        model = env._backend._model
        for _j in range(model.njnt):
            nm = model.joint(_j).name
            if nm and "hip_roll" in nm:
                model.jnt_range[_j] = [0.0, 0.0]
                _jd = int(np.asarray(model.jnt_dofadr)[_j])
                if 0 <= _jd < model.nv:
                    model.dof_damping[_jd] = 100.0
    return env


def read_sensors(env: Any) -> dict[str, Any]:
    """单 env 传感器包."""
    return {
        k: v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v
        for k, v in read_sensors_batch(env).items()
    }


def read_sensors_batch(env: Any) -> dict[str, np.ndarray]:
    """向量化传感器包 (N envs), 底层 API 与经典轨一致."""
    backend = env._backend
    up = np.asarray(backend.get_sensor_data("upvector"), dtype=np.float64)
    gyro = np.asarray(backend.get_sensor_data("gyro"), dtype=np.float64)
    linvel = np.asarray(backend.get_sensor_data("local_linvel"), dtype=np.float64)
    base_pos = np.asarray(backend.get_base_pos(), dtype=np.float64)
    dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float64)
    dof_vel = np.asarray(env.get_dof_vel(), dtype=np.float64)
    L0, theta0 = _fk_legs_batch(dof_pos)
    wheel_z = base_pos[:, 2][:, None] + HIP_Z_OFFSET - L0 * np.abs(np.cos(theta0))
    return {
        "theta": np.arctan2(up[:, 0], up[:, 2]),
        "theta_dot": gyro[:, 1],
        "v": linvel[:, 0],
        "v_wheel": dof_vel[:, 6] * WHEEL_R,
        "omega_z": gyro[:, 2],
        "base_z": base_pos[:, 2],
        "wheel_z": wheel_z[:, 0],
        "base_pos": base_pos,
        "up": up,
        "gyro": gyro,
        "linvel": linvel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
    }
