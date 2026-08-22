"""Smoke tests for the xqrobotwl PPO+VMC and SRL+VMC jump environments."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common.rewards import RewardContext, run_reward_dispatch
from unilab.envs.locomotion.xqrobotwl.jump_srl import (
    _reward_anti_drift,
    _reward_height_progress,
)
from unilab.envs.locomotion.xqrobotwl.jump_vmc import _latch_jump_request
from unilab.envs.locomotion.xqrobotwl.vmc import VirtualLegVMC, XqRobotWLVMCConfig

# Registered env names (see conf/ppo/task/*_vmc_flat/mujoco.yaml)
VMC_ENVS = ("XqRobotWLJumpVMC", "XqRobotWLJumpSRLVMC")

_REWARD = {
    "scales": {
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 1.0,
        "lin_vel_z": -0.2,
        "ang_vel_xy": -0.05,
        "base_height": -60.0,
        "orientation": -5.0,
        "joint_action_rate": -0.1,
        "wheel_action_rate": -0.015,
        "leg_mirror": -12.0,
        "tsk": -2.0,
        "alive": 1.0,
        "jump_height": 12.0,
        "crouch_prep": 4.0,
        "landing_soft": 15.0,
        "wheel_air_time": 20.0,
        "vertical_thrust": 30.0,
        "crouch_depth": 4.0,
        "anti_loiter": 12.0,
        "lean_forward": 5.0,
    },
    "tracking_sigma": 0.3,
    "base_height_target": 0.55,
    "jump_height_target": 1.0,
    "crouch_height_target": 0.40,
    "max_tilt_deg": 60.0,
    "min_base_height": 0.15,
    "jump_curriculum_start": 0,
    # 0 = jump curriculum fully on immediately (matches the SRL+YAML; the pure
    # VMC task uses a longer warm-up in its own YAML, which is irrelevant here).
    "jump_curriculum_end": 0,
}

_DR = {
    "randomize_base_mass": False,
    "randomize_ground_friction": False,
    "randomize_kp": False,
    "randomize_kd": False,
    "random_com": False,
    "randomize_leg_length": False,
}

# Default posture from the scene keyframe (dof order).
_DEFAULT_DOF_POS = np.array(
    [0.1, 0.15, 0.15, -0.1, -0.15, -0.15, 0.0, 0.0], dtype=get_global_dtype()
)


def _vmc_layer(num_envs: int = 2) -> VirtualLegVMC:
    return VirtualLegVMC(XqRobotWLVMCConfig(), num_envs, dtype=get_global_dtype())


# --------------------------------------------------------------------------- #
# Pure unit tests (no MuJoCo required)                                        #
# --------------------------------------------------------------------------- #


def test_fk_round_trip_at_default_posture() -> None:
    vmc = _vmc_layer(2)
    dof_pos = np.broadcast_to(_DEFAULT_DOF_POS, (2, 8)).copy()
    dof_vel = np.zeros((2, 8), dtype=get_global_dtype())
    _, _, theta0, L0, _, _ = vmc.compute_kinematics(dof_pos, dof_vel)
    cfg = vmc._cfg
    assert np.allclose(L0, cfg.l0_offset, atol=5e-3), f"L0={L0} != l0_offset={cfg.l0_offset}"
    assert np.allclose(theta0, cfg.theta0_offset, atol=5e-3), (
        f"theta0={theta0} != theta0_offset={cfg.theta0_offset}"
    )


def test_fk_no_nan_across_joint_grid() -> None:
    vmc = _vmc_layer(1)
    rng = np.random.default_rng(0)
    dof_pos = np.zeros((200, 8), dtype=get_global_dtype())
    dof_pos[:, 0] = 0.1
    dof_pos[:, 3] = -0.1
    dof_pos[:, 1] = rng.uniform(-1.047, 2.094, 200)
    dof_pos[:, 2] = rng.uniform(-0.873, 0.873, 200)
    dof_pos[:, 4] = rng.uniform(-2.094, 1.047, 200)
    dof_pos[:, 5] = rng.uniform(-0.873, 0.873, 200)
    dof_vel = rng.uniform(-5, 5, (200, 8))
    theta1, theta2, theta0, L0, theta0_dot, L0_dot = vmc.compute_kinematics(dof_pos, dof_vel)
    assert np.isfinite(L0).all()
    assert np.isfinite(theta0).all()
    assert np.isfinite(theta0_dot).all()
    assert np.isfinite(L0_dot).all()
    # L0 stays within the calibrated achievable range (with slack)
    assert np.all(L0 > 0.10)
    assert np.all(L0 < 0.60)


def test_compute_torques_finite_across_commands() -> None:
    vmc = _vmc_layer(2)
    dof_pos = np.broadcast_to(_DEFAULT_DOF_POS, (2, 8)).copy()
    dof_vel = np.zeros((2, 8), dtype=get_global_dtype())
    for cmd in (np.zeros((2, 8)), np.full((2, 8), 0.5), np.full((2, 8), -0.5)):
        torques = vmc.compute_torques(cmd, dof_pos, dof_vel, sim_dt=0.005)
        assert torques.shape == (2, 8)
        assert np.isfinite(torques).all()
    # Stretch command (L0 action +1 -> L0 ref > current) must keep torques finite
    stretch = np.zeros((2, 8))
    stretch[:, [2, 6]] = 1.0
    torques2 = vmc.compute_torques(stretch, dof_pos, dof_vel, sim_dt=0.005)
    assert np.isfinite(torques2).all()


def _jump_reward_ctx(
    *, vx: float = 0.0, base_z: float = 0.55, prev_max_z: float = 0.55
) -> RewardContext:
    return RewardContext(
        info={
            "commands": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
            "jump_curriculum": 1.0,
            "jump_phase": np.array([1.0]),
            "episode_prev_max_height": np.array([prev_max_z]),
        },
        linvel=np.array([[vx, 0.0, 0.0]]),
        gyro=np.zeros((1, 3)),
        dof_pos=np.zeros((1, 8)),
        num_envs=1,
        base_height=np.array([base_z]),
    )


def test_anti_drift_negative_scale_is_a_penalty() -> None:
    ctx = _jump_reward_ctx(vx=0.35)
    raw = _reward_anti_drift(ctx)
    assert raw[0] > 0.0
    info = {"steps": np.array([0])}
    total = run_reward_dispatch(
        scales={"anti_drift": -3.0},
        fns={"anti_drift": _reward_anti_drift},
        ctx=ctx,
        info=info,
        enable_log=True,
        ctrl_dt=1.0,
    )
    assert total[0] < 0.0
    assert info["log"]["reward/anti_drift"] < 0.0


def test_height_progress_uses_previous_episode_maximum() -> None:
    rising = _reward_height_progress(_jump_reward_ctx(base_z=0.60, prev_max_z=0.55))
    level = _reward_height_progress(_jump_reward_ctx(base_z=0.60, prev_max_z=0.60))
    assert rising[0] > 0.0
    assert level[0] == pytest.approx(0.0)


def test_held_jump_trigger_creates_only_one_request_until_released() -> None:
    fsm = np.array([-1], dtype=np.int32)
    armed = np.array([True])
    pending = np.array([False])
    request, event = _latch_jump_request(fsm, np.array([True]), armed, pending)
    assert request.tolist() == [1.0]
    assert event.tolist() == [True]

    # The FSM accepted the request. Returning to idle while the key remains
    # held must not start a second jump.
    pending[:] = False
    request, event = _latch_jump_request(fsm, np.array([True]), armed, pending)
    assert request.tolist() == [0.0]
    assert event.tolist() == [False]

    _latch_jump_request(fsm, np.array([False]), armed, pending)
    request, event = _latch_jump_request(fsm, np.array([True]), armed, pending)
    assert request.tolist() == [1.0]
    assert event.tolist() == [True]


def test_wheel_integral_reset() -> None:
    vmc = _vmc_layer(3)
    dof_pos = np.broadcast_to(_DEFAULT_DOF_POS, (3, 8)).copy()
    dof_vel = np.zeros((3, 8), dtype=get_global_dtype())
    policy_ctrl = np.zeros((3, 8), dtype=get_global_dtype())
    policy_ctrl[:, [3, 7]] = 5.0  # strong wheel command accumulates integral
    for _ in range(10):
        vmc.compute_torques(policy_ctrl, dof_pos, dof_vel, sim_dt=0.005)
    assert np.abs(vmc._wheel_integral).sum() > 0
    vmc.reset_wheel_integral(np.array([1], dtype=np.int64))
    assert np.allclose(vmc._wheel_integral[1], 0.0)
    assert np.abs(vmc._wheel_integral[[0, 2]]).sum() > 0


def test_apply_action_reorders_to_actuator_order() -> None:
    """Zero policy action must yield the physical default refs in actuator order."""
    from unilab.base.np_env import NpEnvState
    from unilab.envs.locomotion.xqrobotwl.jump_vmc import (
        XqRobotWLJumpVMCFlatCfg,
        XqRobotWLJumpVMCFlatEnv,
    )

    cfg = XqRobotWLJumpVMCFlatCfg(vmc=XqRobotWLVMCConfig(), reward_config=None)
    # Build a minimal state for apply_action without a backend.
    state = NpEnvState(
        obs={},
        reward=np.zeros(1),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
        info={"current_actions": np.zeros((1, 8))},
    )
    env = object.__new__(XqRobotWLJumpVMCFlatEnv)
    env._cfg = cfg
    env._np_dtype = get_global_dtype()
    env._vmc_cfg = cfg.vmc
    ctrl = XqRobotWLJumpVMCFlatEnv.apply_action(env, np.zeros((1, 8)), state)
    v = cfg.vmc
    assert np.allclose(ctrl[0, 0], v.roll_default[0])  # roll_L
    assert np.allclose(ctrl[0, 1], v.theta0_offset)  # theta_L
    assert np.allclose(ctrl[0, 2], v.l0_offset)  # L0_L
    assert np.allclose(ctrl[0, 3], 0.0)  # wheel_L
    assert np.allclose(ctrl[0, 4], v.roll_default[1])  # roll_R
    assert np.allclose(ctrl[0, 5], v.theta0_offset)  # theta_R
    assert np.allclose(ctrl[0, 6], v.l0_offset)  # L0_R
    assert np.allclose(ctrl[0, 7], 0.0)  # wheel_R


# --------------------------------------------------------------------------- #
# Real-sim smoke tests (MuJoCo required)                                       #
# --------------------------------------------------------------------------- #


def _make_env(name: str, num_envs: int = 2):
    mujoco = pytest.importorskip("mujoco")
    del mujoco
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    return registry.make(
        name,
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override={"reward_config": _REWARD, "domain_rand": _DR},
    )


@pytest.mark.parametrize("env_name", VMC_ENVS)
def test_env_build_reset_step(env_name: str) -> None:
    env = _make_env(env_name)
    try:
        spec = env.obs_groups_spec
        assert spec["obs"] > 0 and spec["critic"] > 0
        obs, _ = env.reset(np.array([0, 1], dtype=np.int32))
        assert obs["obs"].shape == (2, spec["obs"])
        assert obs["critic"].shape == (2, spec["critic"])
        state = env.step(np.random.uniform(-1, 1, (2, 8)))
        assert np.isfinite(state.reward).all()
        assert all(np.isfinite(v).all() for v in state.obs.values())
        torques = env._last_vmc_ctrl
        assert np.isfinite(torques).all()
        assert (torques >= env._ctrl_lower).all() and (torques <= env._ctrl_upper).all()
    finally:
        env.close()


def test_vmc_obs_dims_match_spec() -> None:
    env = _make_env("XqRobotWLJumpVMC")
    try:
        # v9: 干净消融 — PPO+VMC 观测对齐纯PPO (无 SLIP 参考, 无 FSM/虚拟腿特征).
        assert env.obs_groups_spec == {"obs": 297, "critic": 324}
    finally:
        env.close()


def test_srl_vmc_obs_dims_match_spec() -> None:
    env = _make_env("XqRobotWLJumpSRLVMC")
    try:
        # v8: 干净消融 — SRL+VMC 观测对齐 SRL (关节空间 33*9 + 18 = 315),
        # 只输出层 (VMC) 与 SRL 不同。
        assert env.obs_groups_spec == {"obs": 315, "critic": 342}
    finally:
        env.close()


def test_srl_vmc_fsm_advances_on_trigger() -> None:
    env = _make_env("XqRobotWLJumpSRLVMC", num_envs=2)
    try:
        env.reset(np.array([0, 1], dtype=np.int32))
        env.step(np.zeros((2, 8)))  # initialise env state
        # Force a jump trigger; with the curriculum fully on it persists and the
        # contact-driven FSM leaves IDLE after ~0.1 s of held trigger.
        env.state.info["commands"][:, 4] = 1.0
        left_idle = None
        for i in range(30):
            env.step(np.zeros((2, 8)))
            if (env._fsm_state != -1).any():
                left_idle = i
                break
        assert left_idle is not None, "SRL+VMC FSM should leave IDLE on a held trigger"
    finally:
        env.close()


def test_srl_vmc_exposes_previous_height_and_independent_contacts() -> None:
    env = _make_env("XqRobotWLJumpSRLVMC", num_envs=2)
    try:
        env.reset(np.array([0, 1], dtype=np.int32))
        state = env.step(np.zeros((2, 8)))
        assert "episode_prev_max_height" in state.info
        assert state.info["episode_prev_max_height"].shape == (2,)
        assert state.info["wheel_contact"].shape == (2, 2)
        env._backend.get_sensor_data("right_wheel_world_pos")
    finally:
        env.close()


def test_srl_vmc_knee_limit_penalty_and_termination() -> None:
    env = _make_env("XqRobotWLJumpSRLVMC", num_envs=1)
    try:
        ctx_safe = _jump_reward_ctx()
        ctx_near = _jump_reward_ctx()
        ctx_near.dof_pos[0, [2, 5]] = [0.85, -0.85]
        assert env._reward_knee_limit(ctx_safe)[0] == pytest.approx(0.0)
        assert env._reward_knee_limit(ctx_near)[0] > 0.0

        gravity = np.array([[0.0, 0.0, 1.0]])
        dof_pos = np.zeros((1, 8))
        dof_pos[0, 2] = 0.91
        assert env._compute_terminated(gravity, dof_pos)[0]
    finally:
        env.close()


def test_srl_vmc_phase_floors_override_unsafe_policy_residual() -> None:
    env = _make_env("XqRobotWLJumpSRLVMC", num_envs=1)
    try:
        env.init_state()
        env._fsm_state[:] = 0
        state = env.step(np.full((1, 8), -1.0))
        cfg = env._vmc_cfg
        crouch_floor = (cfg.crouch_min_length - cfg.l0_offset) / cfg.action_scale_l0
        assert state.info["current_actions"][0, 2] >= crouch_floor - 1.0e-6
        assert state.info["current_actions"][0, 5] >= crouch_floor - 1.0e-6
    finally:
        env.close()
