"""xqrobotwl SRL+VMC jump env.

Same SLIP-FSM leg-length reference as ``XqRobotWLJumpVMCFlatEnv``, but the
reference is applied in RESIDUAL mode:

    final_L0_action = phase_reference_action + feedback_gain * policy_L0_action

(``feedback_gain`` ~ 0.15 keeps the policy as a small residual on top of the
proven SLIP-FSM trajectory, so it cannot cancel the reference).  Uses the SRL
reward set (with ``height_progress`` etc.) inherited from ``jump_srl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.base import registry
from unilab.envs.locomotion.common import rewards

from . import jump_srl as _srl
from .jump_vmc import XqRobotWLJumpVMCFlatCfg, XqRobotWLJumpVMCFlatEnv


@registry.envcfg("XqRobotWLJumpSRLVMC")
@dataclass
class XqRobotWLJumpSRLVMCFlatCfg(XqRobotWLJumpVMCFlatCfg):
    reward_config: _srl.XqRobotWLJumpRewardConfig | None = None  # type: ignore[assignment]


@registry.env("XqRobotWLJumpSRLVMC", sim_backend="mujoco")
class XqRobotWLJumpSRLVMCFlatEnv(XqRobotWLJumpVMCFlatEnv):
    """SRL + VMC jump -- SLIP-FSM reference applied as a policy residual."""

    _cfg: XqRobotWLJumpSRLVMCFlatCfg

    def __init__(self, cfg: XqRobotWLJumpSRLVMCFlatCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._init_reward_functions()

    def step(self, actions):
        actions = np.asarray(actions, dtype=self._np_dtype).copy()
        target_action = self._jump_leg_reference_action()
        residual_scale = getattr(self._jump_cfg, "feedback_gain", 0.15)
        # Residual mode: the policy provides a small residual on top of the
        # SLIP-FSM reference (reference-project mode), so it cannot cancel the
        # proven crouch-thrust trajectory.
        actions[:, 2] = target_action + residual_scale * actions[:, 2]  # L0_L
        actions[:, 5] = target_action + residual_scale * actions[:, 5]  # L0_R
        return super().step(actions)

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, object] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "joint_action_rate": self._reward_joint_action_rate,
            "wheel_action_rate": self._reward_wheel_action_rate,
            "leg_mirror": self._reward_leg_mirror,
            "tsk": self._reward_tsk,
            "alive": rewards.alive,
            "jump_height": self._reward_jump_height,
            "crouch_prep": self._reward_crouch_prep,
            "landing_soft": self._reward_landing_soft,
            "wheel_air_time": self._reward_wheel_air_time,
            "vertical_thrust": self._reward_vertical_thrust,
            "crouch_depth": self._reward_crouch_depth,
            "lean_forward": self._reward_lean_forward,
            "height_progress": self._reward_height_progress,
        }

    def _reward_jump_height(self, ctx):
        return _srl._reward_jump_height(ctx, self._jump_cfg)

    def _reward_crouch_prep(self, ctx):
        return _srl._reward_crouch_prep(ctx, self._jump_cfg)

    def _reward_landing_soft(self, ctx):
        return _srl._reward_landing_soft(ctx)

    def _reward_wheel_air_time(self, ctx):
        return _srl._reward_wheel_air_time(ctx)

    def _reward_vertical_thrust(self, ctx):
        return _srl._reward_vertical_thrust(ctx, self._jump_cfg)

    def _reward_crouch_depth(self, ctx):
        return _srl._reward_crouch_depth(ctx, self._jump_cfg)

    def _reward_lean_forward(self, ctx):
        return _srl._reward_lean_forward(ctx)

    def _reward_height_progress(self, ctx):
        return _srl._reward_height_progress(ctx)
