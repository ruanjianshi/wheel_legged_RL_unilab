"""xqrobotwl PPO+VMC jump env with SLIP-FSM leg-length reference.

Same phase-gated jump rewards, commands, curriculum and termination as the pure
PPO ``XqRobotWLJumpFlat`` env -- but the 8D policy action is interpreted in the
virtual-leg space (hip-roll reference + virtual leg angle/length + wheel speed)
and realised as joint torques through the VMC layer (see ``vmc.py``).

The SLIP-FSM provides a per-phase virtual-leg-length reference (crouch -> thrust
-> flight retract -> landing absorb).  This env applies it in FULL-ACTION mode:

    final_L0_action = phase_reference_action + policy_L0_action

(the policy keeps full authority on top of the reference, distinct from the
residual mode used by ``XqRobotWLJumpSRLVMCFlatEnv``).

Action layout (dof order, matching the pure-PPO policy space so the reward
functions stay valid):
    [roll_L, theta_L, L0_L, roll_R, theta_R, L0_R, wheel_L, wheel_R]

Observation frame (41 base + 18 FSM features = 387 / critic 486).
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
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._np_dtype = get_global_dtype()

        # Torque limits from the backend actuator ctrl-range (go2w pattern).
        ctrl_range = np.asarray(self._backend.get_actuator_ctrl_range(), dtype=self._np_dtype)
        self._ctrl_lower = ctrl_range[:, 0].astype(self._np_dtype)
        self._ctrl_upper = ctrl_range[:, 1].astype(self._np_dtype)

        self._vmc = VirtualLegVMC(cfg.vmc, num_envs, dtype=self._np_dtype)
        self._last_vmc_ctrl = np.zeros((num_envs, NUM_ACTIONS), dtype=self._np_dtype)

        # Extend observation frames with virtual-leg state + torques.
        self._obs_frame_dim = 41
        self._critic_frame_dim = 52
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )

        self._backend.set_pre_step_control(self._pre_step_vmc_control)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        fsm_extra = 2 * self._hist_len  # 2 FSM features tiled across history
        return {
            "obs": self._obs_frame_dim * self._hist_len + fsm_extra,
            "critic": self._critic_frame_dim * self._hist_len + fsm_extra,
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
        actions = np.asarray(actions, dtype=self._np_dtype).copy()
        target_action = self._jump_leg_reference_action()
        # Full-action mode: the policy keeps full authority on top of the SLIP-FSM
        # leg-length reference (distinct from VMC+SRL's residual mode).
        actions[:, 2] = target_action + actions[:, 2]  # L0_L
        actions[:, 5] = target_action + actions[:, 5]  # L0_R
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
        # Start from the VMC layer's default gains, then scale by FSM phase.
        kp, kd, ff = self._vmc.get_l0_control_parameters()
        cfg = self._vmc_cfg
        phase = self._fsm_state
        thrust = (phase == 1)[:, None]
        flight = (phase == 2)[:, None]
        landing = (phase == 3)[:, None]
        kp = np.where(thrust, kp * cfg.thrust_kp_scale, kp)
        # Thrust uses low leg damping (kd_l0 * thrust_kd_scale) so the extension
        # is explosive enough to break wheel contact -> real lift-off.
        kd = np.where(thrust, kd * cfg.thrust_kd_scale, kd)
        ff = np.where(thrust, ff * cfg.thrust_ff_scale, ff)
        ff = np.where(flight, ff * cfg.flight_ff_scale, ff)
        kd = np.where(landing, kd * cfg.landing_kd_scale, kd)
        ff = np.where(landing, ff * cfg.landing_ff_scale, ff)
        return kp, kd, ff

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
        jt = state.info["commands"][:, 4]
        vmc_cfg = self._vmc_cfg
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
        self._episode_max_height = np.maximum(self._episode_max_height, base_z)
        state.info["episode_max_height"] = self._episode_max_height.copy()
        self._update_wheel_contact(state.info)
        self._update_jump_air_progress(state.info, base_z)
        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        theta1, theta2, theta0, L0, theta0_dot, L0_dot = self._vmc.compute_kinematics(
            dof_pos, dof_vel
        )
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_ACTIONS)))

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_theta0 = self._obs_noise(theta0, noise_cfg.scale_joint_angle)
        noisy_theta0_dot = self._obs_noise(theta0_dot, noise_cfg.scale_joint_vel)
        noisy_L0 = self._obs_noise(L0, noise_cfg.scale_joint_angle)
        noisy_L0_dot = self._obs_noise(L0_dot, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)

        obs_frame = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_leg_diff,
                noisy_leg_vel,
                noisy_theta0,
                noisy_theta0_dot,
                noisy_L0,
                noisy_L0_dot,
                noisy_wheel_vel,
                last_actions,
                info["commands"],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        torques = info.get("torques", np.zeros((linvel.shape[0], NUM_ACTIONS)))
        critic_frame = np.concatenate(
            [
                gyro,
                -gravity,
                leg_diff,
                leg_vel,
                theta0,
                theta0_dot,
                L0,
                L0_dot,
                wheel_vel,
                last_actions,
                info["commands"],
                linvel,
                torques,
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

        obs = self._obs_history[:batch_size].reshape(batch_size, -1)
        critic = self._critic_history[:batch_size].reshape(batch_size, -1)

        # Append FSM features (state + timer) tiled across the history frames.
        fsm_feat = self._fsm_state.astype(np.float64).reshape(-1, 1)[:batch_size] / 5.0
        timer_feat = np.clip(self._fsm_timer.reshape(-1, 1)[:batch_size] / 0.8, 0, 1)
        extra = np.tile(
            np.concatenate([fsm_feat, timer_feat], axis=1, dtype=get_global_dtype())[:, None, :],
            (1, self._hist_len, 1),
        ).reshape(batch_size, -1)
        obs = np.concatenate([obs, extra], axis=1)
        critic = np.concatenate([critic, extra], axis=1)
        return {"obs": obs, "critic": critic}
