"""Regression tests for the single-wheel balance-and-drive task."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.xqrobotwl.single_leg_unicycle import (
    _BASE_Z,
    _mode_progress,
    _reward_lateral_drift,
    _reward_stop_hold,
    _reward_tracking_vx,
    _update_mode_fsm,
)

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml"


def _ctx(
    *, command: float, vx: float, vy: float = 0.0, x: float = 0.0, y: float = 0.0
) -> RewardContext:
    return RewardContext(
        info={
            "commands": np.array([[command, 0.0, 0.0, 0.0, _BASE_Z]]),
            "base_xy": np.array([[x, y]]),
            "current_actions": np.zeros((1, 8)),
            "last_actions": np.zeros((1, 8)),
        },
        linvel=np.array([[vx, vy, 0.0]]),
        gyro=np.zeros((1, 3)),
        dof_pos=np.zeros((1, 6)),
        num_envs=1,
        tracking_sigma=0.04,
    )


def test_tracking_vx_prefers_commanded_velocity() -> None:
    matched = _reward_tracking_vx(_ctx(command=0.2, vx=0.2))[0]
    wrong = _reward_tracking_vx(_ctx(command=0.2, vx=-0.2))[0]
    assert matched == pytest.approx(1.0)
    assert matched > wrong


def test_forward_motion_is_not_classified_as_drift() -> None:
    moving = _ctx(command=0.2, vx=0.2, x=2.0, y=0.0)
    lateral = _ctx(command=0.2, vx=0.2, x=0.0, y=0.3)
    assert _reward_lateral_drift(moving)[0] == pytest.approx(0.0)
    assert _reward_lateral_drift(lateral)[0] > 0.0
    assert _reward_stop_hold(moving)[0] == pytest.approx(0.0)


def _make_env(num_envs: int = 8, reward_override: dict | None = None):
    pytest.importorskip("mujoco")
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    cfg = yaml.safe_load(CONFIG.read_text())
    if reward_override:
        cfg["reward"].update(reward_override)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg["env"])
    return registry.make(
        "XqRobotWLSingleLegUnicycle",
        num_envs=num_envs,
        sim_backend="mujoco",
        env_cfg_override=override,
    )


def test_env_starts_with_small_velocity_curriculum_and_compatible_obs() -> None:
    np.random.seed(7)
    env = _make_env()
    try:
        obs, info = env.reset(np.arange(8, dtype=np.int32))
        assert env.obs_groups_spec == {"obs": 324, "critic": 351}
        assert obs["obs"].shape == (8, 324)
        assert np.max(np.abs(info["commands"][:, 0])) <= 0.030001
        state = env.step(np.zeros((8, 8)))
        assert np.isfinite(state.reward).all()
        assert state.info["command_speed_limit"] > 0.03
        env._backend.get_sensor_data("right_wheel_world_pos")
    finally:
        env.close()


def test_env_can_restore_curriculum_progress_when_resuming() -> None:
    env = _make_env(
        num_envs=1,
        reward_override={
            "command_curriculum_steps": 1000,
            "command_curriculum_start_steps": 400,
        },
    )
    try:
        assert env._command_curriculum_progress() == pytest.approx(0.4)
    finally:
        env.close()


def test_motion_features_include_measured_velocity_without_changing_dimension() -> None:
    env = _make_env(num_envs=1)
    try:
        env.init_state()
        env.state.info["commands"][0, 0] = 0.12
        state = env.step(np.zeros((1, 8)))
        newest_frame = state.obs["obs"][0, -36:]
        features = newest_frame[-8:-3]
        assert features[0] == pytest.approx(0.12, abs=1e-5)
        assert features[3] == pytest.approx(features[0] - features[1], abs=1e-5)
        assert features[4] == pytest.approx(0.518, abs=1e-5)
        env._mode_state[:] = 1
        env._mode_progress[:] = 1.0
        state = env.step(np.zeros((1, 8)))
        assert state.obs["obs"][0, -8:-3][-1] == pytest.approx(_BASE_Z, abs=1e-5)
    finally:
        env.close()


def test_positive_velocity_command_generates_negative_wheel_reference() -> None:
    env = _make_env(num_envs=1)
    try:
        env.init_state()
        env._mode_state[:] = 1
        env._mode_progress[:] = 1.0
        env.state.info["commands"][0, 0] = 0.1
        captured = {}
        original = env.apply_action

        def capture(actions, state):
            captured["actions"] = np.asarray(actions).copy()
            return original(actions, state)

        env.apply_action = capture
        env.step(np.zeros((1, 8)))
        assert captured["actions"][0, 6] < 0.0
    finally:
        env.close()


def test_unicycle_mode_is_latched_until_h_key_is_released() -> None:
    state = np.array([-1], dtype=np.int32)
    timer = np.zeros(1)
    state, timer = _update_mode_fsm(state, timer, np.array([1.0]), 0.01, 1.0)
    assert state.tolist() == [0]
    for _ in range(100):
        state, timer = _update_mode_fsm(state, timer, np.array([1.0]), 0.01, 1.0)
    assert state.tolist() == [1]
    state, timer = _update_mode_fsm(state, timer, np.array([0.0]), 0.01, 1.0)
    assert state.tolist() == [2]


def test_mode_progress_is_smooth_and_reversible() -> None:
    state = np.array([-1, 0, 0, 1, 2, 2])
    timer = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    progress = _mode_progress(state, timer, transition_time=2.0)
    assert progress.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0, 0.5])


def test_final_task_resets_in_normal_two_wheel_standing() -> None:
    env = _make_env(num_envs=1)
    try:
        env.init_state()
        up = np.asarray(env._backend.get_sensor_data("upvector"))[0]
        assert env._mode_state.tolist() == [-1]
        assert up[2] > 0.95
        assert env.state.info["commands"][0, 4] in (0.0, 1.0)
    finally:
        env.close()


def test_reverse_curriculum_can_reset_on_supported_transition_manifold() -> None:
    np.random.seed(11)
    env = _make_env(
        num_envs=4,
        reward_override={
            "transition_reset_probability": 1.0,
            "transition_reset_min_progress": 0.5,
        },
    )
    try:
        env.init_state()
        left_z = np.asarray(env._backend.get_sensor_data("left_wheel_world_pos"))[:, 2]
        assert np.all(env._mode_state == 0)
        assert np.all(env._mode_progress > 0.0)
        assert np.allclose(left_z, 0.11, atol=2.0e-3)
        assert np.isfinite(env.state.obs["obs"]).all()
    finally:
        env.close()
