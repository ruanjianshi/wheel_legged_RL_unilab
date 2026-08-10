"""xqrobotwl 改进结构单轮平衡 RL env — 横躺独轮车式 (从零训练).

改进结构 (devlog 20, 物理可行):
  - 机身横躺 up=[0,-1,0], **左腿近直支撑** L=(1.49,-0.1,-0.2), CoM 偏移 0.5cm
  - **右腿伸直配重** R=(-1.50,0,0), CoM 权限 ±5cm
  - 执行器 (env init 运行时设置): 腿 kp=300 kv=10 (防膝塌陷), 轮=扭矩源
  - 控制通道: 左轮扭矩控 pitch+移动, 右髋 roll/pitch 配重控 roll, 支撑腿钉住

经典控制验证 (devlog 20): PD+滤波 0.86s 才倒 (原结构 0.2s), 剩余=配重 roll
耦合 pitch 的 3D 控制 → 交给 RL 学.

关键观测: 横躺姿态下 pitch (绕轮轴=世界y) 在 up 向量不可见 (up∥轮轴),
必须加 base 长轴向量 (basexvector sensor) 到 obs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.xqrobotwl.base import NUM_LEG_ACTIONS
from unilab.envs.locomotion.xqrobotwl.joystick import (
    XqRobotWLDRProvider,
    XqRobotWLRewardConfig,
    XqRobotWLWalkFlatCfg,
    XqRobotWLWalkFlatEnv,
)

# ── 改进结构参数 (devlog 20 结构搜索) ──
_SUPPORT_POSE = np.array([1.49, -0.1, -0.2], dtype=np.float64)  # 支撑腿(左) L_roll/L_pitch/L_knee
_CW_ROLL0 = -1.50  # 配重 R_hip_roll 基准 (伸直腿)
_CW_PITCH0 = 0.0
_CW_KNEE0 = 0.0
# 横躺参考 (up=base+z 指向 -y, base+x 指向 +x)
_UP_REF = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_XVEC_REF = np.array([1.0, 0.0, 0.0], dtype=np.float64)
# 动作映射
_WHEEL_TORQUE_SCALE = 8.0  # RL 动作 a6 ∈ ~[-1,1] → 轮扭矩 Nm
_WHEEL_TORQUE_LIMIT = 20.0
_RROLL_SCALE = 0.5  # RL 动作 → R_hip_roll 增量 (rad)
_RPITCH_SCALE = 0.5
# reset 几何 (FK 标定): 横躺位姿下左轮贴地的 base 高度
_BASE_Z = 0.6776
_WHEEL_R = 0.11
_NUM_CMD_DIM = 5
# obs 帧: gyro(3)+grav(3)+basex(3)+leg_diff(6)+leg_vel(6)+wheel_vel(2)+act(8)+cmd(5)
_OBS_FRAME_DIM = 36
_CRITIC_FRAME_DIM = 39
# 位姿默认角 (leg_diff 参考) — 支撑腿钉在 pose, 配重基准
_POSE_DEFAULT = np.array([1.49, -0.1, -0.2, -1.50, 0.0, 0.0], dtype=np.float64)
# 终止阈值
_WHEEL_LIFT_THRESHOLD = 0.02  # 左轮 z 超过贴地高度 2cm = 离地
_ALIGN_BAD = 0.80  # up 或 basex 对齐度低于此 = 大幅倾覆
_MIN_BASE_Z = 0.30


@dataclass
class XqRobotWLSingleLegUnicycleCommands(Commands):
    # 5D: [vx, vy, vyaw, tsk, height] — 第一阶段只保持 (vx=0), 第二阶段开 vx
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0, _BASE_Z], [0.0, 0.0, 0.0, 0.0, _BASE_Z]]
    )
    resampling_time: float = 3.0


@dataclass
class XqRobotWLSingleLegUnicycleRewardConfig(XqRobotWLRewardConfig):
    scales: dict[str, float] = field(default_factory=dict)
    tracking_sigma: float = 0.3
    base_height_target: float = _BASE_Z
    only_positive_rewards: bool = False
    max_tilt_deg: float = 45.0
    min_base_height: float = _MIN_BASE_Z


@registry.envcfg("XqRobotWLSingleLegUnicycle")
@dataclass
class XqRobotWLSingleLegUnicycleCfg(XqRobotWLWalkFlatCfg):
    commands: XqRobotWLSingleLegUnicycleCommands = field(
        default_factory=XqRobotWLSingleLegUnicycleCommands
    )
    reward_config: XqRobotWLSingleLegUnicycleRewardConfig | None = None
    max_episode_seconds: float = 8.0


# ── 奖励 ──


def _reward_upright(ctx: RewardContext) -> np.ndarray:
    """机身保持横躺参考: up=[0,-1,0] (roll) 与 base长轴=[1,0,0] (pitch) 同时对齐。"""
    up = np.asarray(ctx.gravity)[: ctx.num_envs]
    xvec = np.asarray(ctx.info.get("basex", np.tile(_XVEC_REF, (ctx.num_envs, 1))))
    roll_a = up @ _UP_REF
    pitch_a = xvec @ _XVEC_REF
    r = np.clip((roll_a - 0.9) / 0.1, 0.0, 1.0) * np.clip((pitch_a - 0.9) / 0.1, 0.0, 1.0)
    return r**2


def _reward_wheel_down(ctx: RewardContext) -> np.ndarray:
    """支撑轮(左)贴地: z 接近 WHEEL_R 给奖。"""
    lw_z = np.asarray(ctx.info.get("left_wheel_z", np.full(ctx.num_envs, _WHEEL_R)))
    err = np.abs(lw_z - _WHEEL_R)
    return np.clip((0.03 - err) / 0.03, 0.0, 1.0)


def _reward_counterweight_pose(ctx: RewardContext) -> np.ndarray:
    """配重姿态: R_roll≈-1.50, R_pitch≈0, R_knee≈0 (伸直)。"""
    dp = ctx.dof_pos[: ctx.num_envs, 3:6]
    r_roll = dp[:, 0] - _CW_ROLL0
    r_pitch = dp[:, 1] - _CW_PITCH0
    r_knee = dp[:, 2] - _CW_KNEE0
    return np.sum(np.square(np.stack([r_roll, r_pitch, r_knee], axis=1)), axis=1)


def _reward_damping(ctx: RewardContext) -> np.ndarray:
    """角速度阻尼: 罚 gyro 幅度 (保持静止平衡)。"""
    gyro = ctx.gyro[: ctx.num_envs]
    return np.sum(np.square(gyro), axis=1)


def _reward_drift(ctx: RewardContext) -> np.ndarray:
    """净漂移惩罚: 罚 base 水平位移平方 (允许轮子滚动平衡, 只罚最终漂移)。

    ⚠️ 不能用线速度惩罚 — 倒立摆平衡必须让轮子滚动产生速度, 罚速度会阻止平衡。
    """
    xy = np.asarray(ctx.info.get("base_xy", np.zeros((ctx.num_envs, 2))))
    return np.sum(np.square(xy), axis=1)


def _reward_action_rate(ctx: RewardContext) -> np.ndarray:
    cur = ctx.info["current_actions"][:, :8]
    lst = ctx.info["last_actions"][:, :8]
    return np.sum(np.square(cur - lst), axis=1)


class XqRobotWLSingleLegUnicycleDRProvider(XqRobotWLDRProvider):
    """reset 置横躺改进位姿 (start_in_balance), 命令固定。"""

    def _sample_commands(self, env: object, num_reset: int) -> np.ndarray:
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())  # type: ignore[attr-defined]
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())  # type: ignore[attr-defined]
        cmds = np.tile(low, (num_reset, 1))
        cmds[:, 0] = np.random.uniform(low[0], high[0], num_reset)
        return cmds

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        # 横躺 quat = R_x(90°) → up=[0,-1,0]
        qpos = np.tile(
            np.array(
                [
                    0.0,
                    0.0,
                    _BASE_Z,
                    0.70710678,
                    0.70710678,
                    0.0,
                    0.0,
                    _SUPPORT_POSE[0],
                    _SUPPORT_POSE[1],
                    _SUPPORT_POSE[2],
                    0.0,  # 左腿
                    _CW_ROLL0,
                    _CW_PITCH0,
                    _CW_KNEE0,
                    0.0,  # 右腿
                ],
                dtype=get_global_dtype(),
            ),
            (num_reset, 1),
        )
        qvel = np.zeros((num_reset, 14), dtype=get_global_dtype())
        randomization = build_common_reset_randomization(env, num_reset)
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )


@registry.env("XqRobotWLSingleLegUnicycle", sim_backend="mujoco")
class XqRobotWLSingleLegUnicycleEnv(XqRobotWLWalkFlatEnv):
    """改进结构单轮平衡 — 横躺独轮车式 (轮扭矩pitch + 配重roll)。"""

    _cfg: XqRobotWLSingleLegUnicycleCfg

    def __init__(self, cfg: XqRobotWLSingleLegUnicycleCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLSingleLegUnicycleDRProvider()  # type: ignore[union-attr]
        # 重设 obs 帧维度 (加 basex 3D)
        self._obs_frame_dim = _OBS_FRAME_DIM
        self._critic_frame_dim = _CRITIC_FRAME_DIM
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=get_global_dtype()
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=get_global_dtype()
        )
        # 执行器运行时设置: 腿 kp=300 kv=10 (防膝塌陷), 轮=扭矩源 (force=ctrl)
        model = self._backend._model  # type: ignore[attr-defined]
        for a in [0, 1, 2, 4, 5, 6]:
            model.actuator_gainprm[a, 0] = 300.0
            model.actuator_biasprm[a, 1] = -300.0
            model.actuator_biasprm[a, 2] = -10.0
        for a in [3, 7]:
            model.actuator_gainprm[a, 0] = 1.0
            model.actuator_biasprm[a, 2] = 0.0

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._obs_frame_dim * self._hist_len,
            "critic": self._critic_frame_dim * self._hist_len,
        }

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """钉住支撑腿 + 配重基准, RL 控 [R_roll, R_pitch, L_wheel]。

        策略动作序 [L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee,L_wheel,R_wheel]:
          - 支撑腿(L_roll,L_pitch,L_knee) 钉在 _SUPPORT_POSE
          - R_roll/R_pitch 配重: 基准 + RL 增量
          - L_wheel: RL 扭矩 (clipped ±20 Nm)
          - R_knee 钉 0, R_wheel 钉 0
        ctrl (MuJoCo 序) = [L_roll,L_pitch,L_knee,L_wheel,R_roll,R_pitch,R_knee,R_wheel]
        """
        clipped = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=get_global_dtype(),
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped))
        state.info["current_actions"] = clipped
        num = clipped.shape[0]
        ctrl = np.zeros((num, 8), dtype=get_global_dtype())
        ctrl[:, 0] = _SUPPORT_POSE[0]
        ctrl[:, 1] = _SUPPORT_POSE[1]
        ctrl[:, 2] = _SUPPORT_POSE[2]
        ctrl[:, 3] = np.clip(
            clipped[:, 6] * _WHEEL_TORQUE_SCALE, -_WHEEL_TORQUE_LIMIT, _WHEEL_TORQUE_LIMIT
        )
        ctrl[:, 4] = np.clip(_CW_ROLL0 + clipped[:, 3] * _RROLL_SCALE, -3.1416, 0.0)
        ctrl[:, 5] = np.clip(_CW_PITCH0 + clipped[:, 4] * _RPITCH_SCALE, -2.094, 1.047)
        ctrl[:, 6] = _CW_KNEE0
        ctrl[:, 7] = 0.0
        return ctrl

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        n_batch = linvel.shape[0]
        basex = np.asarray(self._backend.get_sensor_data("basexvector"), dtype=get_global_dtype())[
            :n_batch
        ]
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - _POSE_DEFAULT
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_basex = self._obs_noise(basex, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], 8)))

        obs_frame = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_basex,
                noisy_leg_diff,
                noisy_leg_vel,
                noisy_wheel_vel,
                last_actions,
                info["commands"],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        critic_frame = np.concatenate(
            [
                gyro,
                -gravity,
                basex,
                leg_diff,
                leg_vel,
                wheel_vel,
                last_actions,
                info["commands"],
                linvel,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )
        batch_size = obs_frame.shape[0]
        steps_val = int(info.get("steps", np.zeros(1, dtype=np.uint32))[0])
        if steps_val <= 1:
            for i in range(self._hist_len):
                self._obs_history[:batch_size, i, :] = obs_frame
                self._critic_history[:batch_size, i, :] = critic_frame
        else:
            self._obs_history[:batch_size, :-1, :] = self._obs_history[:batch_size, 1:, :]
            self._obs_history[:batch_size, -1, :] = obs_frame
            self._critic_history[:batch_size, :-1, :] = self._critic_history[:batch_size, 1:, :]
            self._critic_history[:batch_size, -1, :] = critic_frame
        return {
            "obs": self._obs_history[:batch_size].reshape(batch_size, -1),
            "critic": self._critic_history[:batch_size].reshape(batch_size, -1),
        }

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        lw_pos = np.asarray(
            self._backend.get_sensor_data("left_wheel_world_pos"), dtype=get_global_dtype()
        )
        lw_z = lw_pos[:, 2]
        wheel_lost = lw_z > _WHEEL_R + _WHEEL_LIFT_THRESHOLD
        basex = self._backend.get_sensor_data("basexvector")
        roll_a = gravity[:, :3] @ _UP_REF
        pitch_a = basex @ _XVEC_REF
        tilted = (roll_a < _ALIGN_BAD) | (pitch_a < _ALIGN_BAD)
        height_bad = base_z < _MIN_BASE_Z
        return np.asarray(wheel_lost | tilted | height_bad, dtype=bool)

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "upright": _reward_upright,
            "wheel_down": _reward_wheel_down,
            "counterweight_pose": _reward_counterweight_pose,
            "damping": _reward_damping,
            "drift": _reward_drift,
            "action_rate": _reward_action_rate,
            "alive": rewards.alive,
        }

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        info["basex"] = np.asarray(self._backend.get_sensor_data("basexvector"), dtype=dtype)
        lw_pos = np.asarray(self._backend.get_sensor_data("left_wheel_world_pos"), dtype=dtype)
        info["left_wheel_z"] = lw_pos[:, 2]
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=dtype)
        info["base_xy"] = base_pos[:, :2]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=_POSE_DEFAULT,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )
