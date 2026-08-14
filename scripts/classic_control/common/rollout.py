"""Env 驱动: 独立任务轨 (registry.make + StandingProvider + 固定命令 + 每步状态录制).

只读复用 RL env/XML/sensor API, 不修改任何 RL 文件。
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.classic_control.common.config import (
    HIP_Z_OFFSET,
    STANDING_ANGLES,
    STANDING_BASE_Z,
    WHEEL_R,
    env_conf,
)
from scripts.classic_control.common.leg_control import fk_legs
from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLDRProvider

ROOT = Path(__file__).resolve().parents[3]


def _env_conf(task_key: str) -> dict:
    """经典控制任务配置的 env 段 (来自本轨 conf/task/<task>.yaml (conf/lqr 或 conf/mpc))."""
    return env_conf(task_key)


def _load_conf(task_key: str) -> dict:
    """加载 RL mujoco.yaml (只读复用, 路径由经典配置 env.mujoco_yaml 指定)."""
    with open(ROOT / _env_conf(task_key)["mujoco_yaml"], encoding="utf-8") as f:
        return yaml.safe_load(f)


class StandingProvider(XqRobotWLDRProvider):
    """固定站姿 provider: 直立 (yaw=0), 零 qvel, 固定命令.

    子类化真实 provider, 只覆写 build_reset_plan; 其余接口 (validate/
    build_reset_observation/build_interval_randomization_plan) 继承.
    rough env 复用 (spawn origins 由 base 逻辑保持).
    """

    def __init__(self, cmd: np.ndarray) -> None:
        self._cmd = np.asarray(cmd, dtype=np.float64)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        qpos = plan.qpos.copy()
        # 自然站姿起步: 直立 (yaw=0), 腿 standing_angles, 轮 0, wheels 着地 (base_z≈0.518)
        qpos[:, 2] = STANDING_BASE_Z
        qpos[:, 3:7] = [1.0, 0.0, 0.0, 0.0]
        qpos[:, 7:15] = [
            *STANDING_ANGLES[:3],
            0.0,
            *STANDING_ANGLES[3:],
            0.0,
        ]
        qvel = np.zeros_like(plan.qvel)
        cmd = np.tile(self._cmd, (len(env_ids), 1))
        info_updates = dict(plan.info_updates)
        info_updates["commands"] = cmd
        return dataclasses.replace(plan, qpos=qpos, qvel=qvel, info_updates=info_updates)


class CmdSchedule:
    """段式命令表: [(时长 s, 5D/4D cmd)]; P1 用单段大时长."""

    def __init__(self, segments: list[tuple[float, list[float]]]) -> None:
        self.segments = [(dur, np.asarray(cmd, dtype=np.float64)) for dur, cmd in segments]

    def at(self, t: float) -> np.ndarray:
        acc = 0.0
        for dur, cmd in self.segments:
            if t < acc + dur:
                return cmd
            acc += dur
        return self.segments[-1][1]


def build_env(
    task_key: str = "walk_flat",
    num_envs: int = 1,
    cmd: np.ndarray | None = None,
    lock_hip_roll: bool = True,
) -> Any:
    """构建确定性经典控制环境 (关课程/域随机化/命令重采样/平滑/噪声).

    lock_hip_roll=True: ★ 锁定髋 roll 自由度 (jnt_range=[0,0]), 腿长变化时防侧向晃动 → 更稳定.
    """
    ensure_registries()
    cfg = _load_conf(task_key)
    override: dict[str, Any] = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))
    override["curriculum"] = {"enabled": False}
    override["max_episode_seconds"] = 1000.0  # ★ 经典控制: 关 episode 截断 (长时评估)
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

    env = registry.make(
        _env_conf(task_key)["task_name"],
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )
    # 固定命令维度: flat 5D / rough 4D (来自经典配置)
    cmd_dim = int(_env_conf(task_key)["command_dim"])
    if cmd is None:
        cmd = np.zeros(cmd_dim, dtype=np.float64)
        if cmd_dim == 5:
            cmd[4] = STANDING_BASE_Z
    env._dr_manager._provider = StandingProvider(cmd[:cmd_dim])  # type: ignore[union-attr]
    env.set_autoreset(False)
    if lock_hip_roll:
        # ★ 锁定髋 roll (jnt_range=[0,0] + 高阻尼) — 消除腿长变化时的侧向晃动
        model = env._backend._model
        for _j in range(model.njnt):
            nm = model.joint(_j).name
            if nm and "hip_roll" in nm:
                model.jnt_range[_j] = [0.0, 0.0]
                _jd = int(np.asarray(model.jnt_dofadr)[_j])
                if 0 <= _jd < model.nv:
                    model.dof_damping[_jd] = 100.0  # 高阻尼: 强阻侧向运动
    return env


def read_sensors(env: Any) -> dict[str, Any]:
    """单 env 传感器包."""
    backend = env._backend
    up = np.asarray(backend.get_sensor_data("upvector"), dtype=np.float64)[0]
    gyro = np.asarray(backend.get_sensor_data("gyro"), dtype=np.float64)[0]
    linvel = np.asarray(backend.get_sensor_data("local_linvel"), dtype=np.float64)[0]
    base_pos = np.asarray(backend.get_base_pos(), dtype=np.float64)[0]
    dof_pos = np.asarray(env.get_dof_pos(), dtype=np.float64)[0]
    dof_vel = np.asarray(env.get_dof_vel(), dtype=np.float64)[0]
    # 轮地面高度 ≈ base_z − 竖直腿长 (FK; 无 track sensor 用运动学)
    L0, theta0 = fk_legs(dof_pos)
    wheel_z = base_pos[2] + HIP_Z_OFFSET - L0 * np.abs(np.cos(theta0))
    return {
        "theta": float(np.arctan2(up[0], up[2])),
        "theta_dot": float(gyro[1]),
        "v": float(linvel[0]),
        "v_wheel": float(
            dof_vel[6] * WHEEL_R
        ),  # ★ 轮线速度 (L 轮 qvel·R; dof 序 [..,L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee,L_wheel,R_wheel])
        "omega_z": float(gyro[2]),
        "base_z": float(base_pos[2]),
        "base_pos": base_pos,
        "up": up,
        "gyro": gyro,
        "linvel": linvel,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "wheel_z": wheel_z,
    }


def run_episode(
    env: Any,
    controller: Any,
    schedule: CmdSchedule,
    sim_time: float,
    dt: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑一段 episode, 返回 (record, stats). 终止即断."""
    env.init_state()
    record: list[dict[str, Any]] = []
    n_steps = int(sim_time / dt)
    terminated = False
    for step in range(n_steps):
        t = step * dt
        cmd = schedule.at(t)
        # ★ 命令按 env 维度裁剪 (flat=5D含height, rough=4D无height)
        cmd = np.asarray(cmd, dtype=np.float64)[: env.state.info["commands"].shape[1]]  # type: ignore[union-attr]
        sensors = read_sensors(env)
        a = controller.act(sensors, cmd)
        # 固定命令注入 (resampling_time=0 不会被覆盖)
        env.state.info["commands"][:] = np.tile(cmd, (env.num_envs, 1))  # type: ignore[union-attr]
        st = env.step(np.asarray(a, dtype=np.float64)[None, :])
        phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        # ★ physics_state = [pad(1), qpos(15), qvel(14)] → 取 [t, qpos, qvel]
        record.append(
            {
                "t": t,
                "state": np.concatenate([[t], phy[1:]]),
                "sensors": sensors,
                "cmd": np.asarray(cmd, dtype=np.float64).copy(),
                "action": np.asarray(a, dtype=np.float64).copy(),
                "terminated": bool(st.terminated[0]),
                "truncated": bool(st.truncated[0]),
            }
        )
        if st.terminated[0] or st.truncated[0]:
            terminated = True
            break
    stats = {
        "ep_len_s": len(record) * dt,
        "terminated": terminated,
        "n_steps": len(record),
    }
    return record, stats
