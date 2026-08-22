"""xqrobotwl PPO+VMC jump env — 干净消融: 无参考, 只输出层与纯PPO 不同.

v9 消融设计 (2×2): 纯PPO vs PPO+VMC 应只在**输出/控制层**不同 —
同样的纯PPO 奖励、同样的 297D 关节观测、同样的训练, 唯一的区别是:
纯PPO 输出关节位置目标 (PD 控制), PPO+VMC 输出虚拟腿参考 (VMC 力控)。

  * **无 SLIP-FSM 参考**: 策略直接输出虚拟腿参考 (roll + θ₀ + L₀ + 轮速),
    VMC 经雅可比映射为关节力矩; 没有参考混合, 没有分阶段 VMC 增益 (恒定)。
  * **奖励**: 与纯PPO 完全相同 (继承 jump.py, 含 launch_rise/jump_height
    phase-gated 版; 无 anti_early_extend — 那是 SLIP 参考时代的产物)。
  * **观测**: 与纯PPO 完全相同 (297D 关节空间, 无虚拟腿/FSM 特征) —
    策略从关节角度推断腿长, 只输出虚拟腿参考。

Action layout (dof order):
    [roll_L, theta_L, L0_L, roll_R, theta_R, L0_R, wheel_L, wheel_R]

SRL+VMC (``XqRobotWLJumpSRLVMCFlatEnv``) 继承本类并在 step/_compute_obs/
get_l0_control_parameters 中覆盖, 加回 SLIP-FSM 参考 — 顺带修复了原
SRL+VMC 的"参考双重混合" (jump_vmc.step 在 SRL+VMC.step 之后再混一次)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.xqrobotwl.base import (
    DEFAULT_ANGLES,
    NUM_ACTIONS,
    NUM_LEG_ACTIONS,
    stack_joint_sensors,
    stack_joint_vel_sensors,
)
from unilab.envs.locomotion.xqrobotwl.jump import (
    XqRobotWLJumpFlatCfg,
    XqRobotWLJumpFlatEnv,
    XqRobotWLJumpRewardConfig,
)
from unilab.envs.locomotion.xqrobotwl.vmc import VirtualLegVMC, XqRobotWLVMCConfig

from . import jump_srl as _srl


def _latch_jump_request(
    fsm_state: np.ndarray,
    raw_trigger: np.ndarray,
    armed: np.ndarray,
    pending: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a held command into one pending request per rising edge."""
    raw = np.asarray(raw_trigger, dtype=bool)
    armed[~raw] = True
    new_request = raw & armed & (fsm_state == -1) & ~pending
    pending[new_request] = True
    armed[new_request] = False
    return pending.astype(np.float64), new_request


@registry.envcfg("XqRobotWLJumpVMC")
@dataclass
class XqRobotWLJumpVMCFlatCfg(XqRobotWLJumpFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotwl" / "scene_flat_vmc.xml")
        )
    )
    vmc: XqRobotWLVMCConfig = field(default_factory=XqRobotWLVMCConfig)
    reward_config: XqRobotWLJumpRewardConfig | None = None  # type: ignore[assignment]


@registry.env("XqRobotWLJumpVMC", sim_backend="mujoco")
class XqRobotWLJumpVMCFlatEnv(XqRobotWLJumpFlatEnv):
    """PPO + VMC jump -- virtual-leg action space + SLIP-FSM reference (full-action)."""

    _cfg: XqRobotWLJumpVMCFlatCfg

    def __init__(self, cfg: XqRobotWLJumpVMCFlatCfg, num_envs=1, backend_type="mujoco"):
        self._vmc_cfg = cfg.vmc
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._episode_max_height = np.zeros(num_envs, dtype=np.float64)
        self._jump_trigger_armed = np.ones(num_envs, dtype=bool)
        self._jump_request_pending = np.zeros(num_envs, dtype=bool)
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._np_dtype = get_global_dtype()

        # Torque limits from the backend actuator ctrl-range (go2w pattern).
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=self._np_dtype)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)

        self._vmc = VirtualLegVMC(cfg.vmc, num_envs, dtype=self._np_dtype)
        self._last_vmc_ctrl = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)

        # v9: PPO+VMC 为无参考消融臂 — 观测对齐纯PPO (33D/帧, 无虚拟腿/FSM 特征)。
        # SRL+VMC 在自己的 __init__ 里按需重设。
        self._obs_frame_dim = 33
        self._critic_frame_dim = 36
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )

        self._backend.set_pre_step_control(self._pre_step_vmc_control)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # v9: 无参考消融臂 — 与纯PPO 相同 (无 FSM 特征): 33*9=297 / 36*9=324。
        return {
            "obs": self._obs_frame_dim * self._hist_len,
            "critic": self._critic_frame_dim * self._hist_len,
        }

    # ------------------------------------------------------------------ #
    # SLIP-FSM leg-length reference                                       #
    # ------------------------------------------------------------------ #

    def _jump_leg_reference(self) -> np.ndarray:
        """Physical L0 reference for the current FSM phase (per env)."""
        cfg = self._vmc_cfg
        target = np.full(self._num_envs, cfg.l0_offset, dtype=self._np_dtype)
        phase = self._fsm_state
        linvel = self.get_local_linvel()
        vz = np.asarray(linvel[:, 2], dtype=self._np_dtype)

        crouch = phase == 0
        target[crouch] = cfg.crouch_length
        thrust = phase == 1
        target[thrust] = cfg.thrust_length
        flight = phase == 2
        if flight.any():
            retract = cfg.flight_retract_length
            denom = max(cfg.prelanding_start_vz - cfg.prelanding_full_vz, 1.0e-6)
            frac = np.clip((cfg.prelanding_start_vz - vz) / denom, 0.0, 1.0)
            target[flight] = retract + frac[flight] * (cfg.prelanding_length - retract)
        landing = phase == 3
        if landing.any():
            frac = np.clip(
                self._fsm_timer[landing] / max(cfg.landing_compression_time, 1.0e-6),
                0.0,
                1.0,
            )
            target[landing] = cfg.prelanding_length + frac * (
                cfg.landing_absorption_length - cfg.prelanding_length
            )
        return target

    def _jump_leg_reference_action(self) -> np.ndarray:
        cfg = self._vmc_cfg
        target = self._jump_leg_reference()
        return np.clip((target - cfg.l0_offset) / cfg.action_scale_l0, -1.0, 1.0)

    def step(self, actions):
        # v9: PPO+VMC 为无参考消融臂 — 策略直接输出虚拟腿参考, VMC 转力矩,
        # 无 SLIP-FSM 参考混合。SRL+VMC 在自己的 step 里做残差参考混合后调用
        # 本方法 (透传) — 顺带修复原 SRL+VMC 的"参考双重混合" (jump_vmc.step
        # 再混一次, 使参考 L0 目标 ×2)。
        return super().step(actions)

    # ------------------------------------------------------------------ #
    # Action -> physical VMC references -> torques                        #
    # ------------------------------------------------------------------ #

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        cfg = self._vmc_cfg
        clipped = np.asarray(
            np.clip(actions, -cfg.clip_actions, cfg.clip_actions), dtype=self._np_dtype
        )
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped))
        state.info["current_actions"] = clipped
        # v7c: 动作一阶低通 (control_config.action_smoothing, 文献 LPF 消高频抖动)
        # VMC 变体原本不走 joystick.apply_action 的 smoothing, 这里补上 —
        # 配合 ang_vel_xy 加强压站立期振荡 (v7b 收敛后站立 |gyro| 2.25 超标)。
        if getattr(self._cfg.control_config, "action_smoothing", 0.0) > 0.0:
            alpha = float(self._cfg.control_config.action_smoothing)
            prev = getattr(self, "_prev_vmc_filtered_action", clipped)
            clipped = alpha * prev + (1.0 - alpha) * clipped
            self._prev_vmc_filtered_action = clipped
        # Policy/dof order: [roll_L, theta_L, L0_L, roll_R, theta_R, L0_R, wheel_L, wheel_R]
        roll_L = clipped[:, 0] * cfg.action_scale_roll + cfg.roll_default[0]
        theta_L = clipped[:, 1] * cfg.action_scale_theta + cfg.theta0_offset
        L0_L = np.clip(clipped[:, 2] * cfg.action_scale_l0 + cfg.l0_offset, cfg.l0_min, cfg.l0_max)
        roll_R = clipped[:, 3] * cfg.action_scale_roll + cfg.roll_default[1]
        theta_R = clipped[:, 4] * cfg.action_scale_theta + cfg.theta0_offset
        L0_R = np.clip(clipped[:, 5] * cfg.action_scale_l0 + cfg.l0_offset, cfg.l0_min, cfg.l0_max)
        wheel_L = clipped[:, 6] * cfg.action_scale_vel * cfg.wheel_sign[0]
        wheel_R = clipped[:, 7] * cfg.action_scale_vel * cfg.wheel_sign[1]
        # Reorder to actuator order [roll_L, theta_L, L0_L, wheel_L, roll_R, theta_R, L0_R, wheel_R].
        return np.stack([roll_L, theta_L, L0_L, wheel_L, roll_R, theta_R, L0_R, wheel_R], axis=1)

    def get_l0_control_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # v9: 无参考消融臂 — 恒定默认增益 (无 SLIP-FSM 分阶段缩放)。
        # SRL+VMC 覆盖此方法保留分阶段增益。
        return self._vmc.get_l0_control_parameters()

    # ------------------------------------------------------------------ #
    # Reward overrides (PPO+VMC-specific, isolated from shared jump.py)    #
    # ------------------------------------------------------------------ #

    def _init_reward_functions(self) -> None:
        # v9: 无参考消融臂 — 奖励与纯PPO 完全相同 (继承 jump.py, 不加
        # anti_early_extend; launch_rise 用 jump.py 的 phase-gated 版)。
        super()._init_reward_functions()

    def _pre_step_vmc_control(self, backend, policy_ctrl: np.ndarray) -> np.ndarray:
        dof_pos = stack_joint_sensors(backend, dtype=self.default_angles.dtype)
        dof_vel = stack_joint_vel_sensors(backend, dtype=self.default_angles.dtype)
        l0_kp, l0_kd, feedforward = self.get_l0_control_parameters()
        torques = self._vmc.compute_torques(
            policy_ctrl,
            dof_pos,
            dof_vel,
            float(self._cfg.sim_dt),
            l0_kp=l0_kp,
            l0_kd=l0_kd,
            feedforward=feedforward,
        )
        np.clip(torques, self._ctrl_lower, self._ctrl_upper, out=torques)
        self._last_vmc_ctrl = torques.copy()
        return torques

    def reset(self, env_indices: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        env_ids = np.asarray(env_indices, dtype=np.int64)
        out = super().reset(env_indices)
        if env_ids.size:
            self._vmc.reset_wheel_integral(env_ids)
            self._last_vmc_ctrl[env_ids] = 0.0
            self._fsm_state[env_ids] = -1
            self._fsm_timer[env_ids] = 0.0
            self._episode_max_height[env_ids] = 0.0
            self._jump_trigger_armed[env_ids] = True
            self._jump_request_pending[env_ids] = False
            if hasattr(self, "_prev_vmc_filtered_action"):
                self._prev_vmc_filtered_action[env_ids] = 0.0
        return out

    # ------------------------------------------------------------------ #
    # Step / obs                                                          #
    # ------------------------------------------------------------------ #

    def update_state(self, state: NpEnvState) -> NpEnvState:
        state.info["torques"] = self._last_vmc_ctrl.copy()
        assert self._jump_cfg is not None
        self._update_commands(state.info)
        # Jump curriculum
        self._total_env_steps += self._num_envs
        progress = np.clip(
            (self._total_env_steps - self._jump_curriculum_start) / self._jump_curriculum_step,
            0.0,
            1.0,
        )
        state.info["commands"][:, 4] *= np.float64(progress)
        state.info["jump_curriculum"] = np.float64(progress)
        # Reset episode_max_height for freshly reset envs
        fresh = state.info["steps"] <= 1
        self._episode_max_height[fresh] = 0.0
        # Jump phase
        trigger_active = state.info["commands"][:, 4] > 0.5
        prev_phase = state.info.get("jump_phase", np.zeros(self._num_envs, dtype=np.float64))
        state.info["jump_phase"] = np.where(trigger_active, prev_phase + 1, np.float64(0))
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        # SLIP FSM
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        jt, new_request = _latch_jump_request(
            self._fsm_state,
            trigger_active,
            self._jump_trigger_armed,
            self._jump_request_pending,
        )
        state.info["jump_request_event"] = new_request.astype(np.float64)
        # A new jump window gets its own height-progress baseline; otherwise
        # the reset pose's height can make height_progress identically zero.
        self._episode_max_height[new_request] = base_z[new_request]
        vmc_cfg = self._vmc_cfg
        previous_fsm = self._fsm_state.copy()
        self._fsm_state, self._fsm_timer = _srl._update_fsm_state(
            self._fsm_state,
            self._fsm_timer,
            base_z,
            linvel,
            dof_pos,
            jt,
            self._jump_cfg.base_height_target,
            self._cfg.ctrl_dt,
            crouch_time=vmc_cfg.fsm_crouch_time,
            thrust_time=vmc_cfg.fsm_thrust_time,
        )
        started = (previous_fsm == -1) & (self._fsm_state == 0)
        self._jump_request_pending[started] = False
        # Height-progress rewards must compare against the maximum from the
        # previous control step.  SRL already exposes this snapshot; keep the
        # VMC inheritance path equivalent so SRL+VMC does not silently receive
        # an all-zero height_progress reward.
        state.info["episode_prev_max_height"] = self._episode_max_height.copy()
        self._episode_max_height = np.maximum(self._episode_max_height, base_z)
        state.info["episode_max_height"] = self._episode_max_height.copy()
        # 几何接触检测 (轮心世界 z < 0.13): 对空中扇腿免疫, 修复 force 阈值法
        # 把腾空误判为着地 → air 门控奖励 (jump_height/wheel_air_time) 从未生效的 bug。
        # 依赖 xqrobotwl_vmc.xml 新增的 left_wheel_world_pos framepos 传感器。
        self._update_wheel_contact_geom(state.info)
        self._update_jump_air_progress(state.info, base_z)
        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        # v9: 无参考消融臂 — 观测与纯PPO 完全相同 (33D/帧, 无虚拟腿/FSM 特征)。
        # 策略从关节角度/速度推断腿长, 只输出层 (虚拟腿参考) 与纯PPO 不同。
        return super(XqRobotWLJumpVMCFlatEnv, self)._compute_obs(
            info, linvel, gyro, gravity, dof_pos, dof_vel
        )
