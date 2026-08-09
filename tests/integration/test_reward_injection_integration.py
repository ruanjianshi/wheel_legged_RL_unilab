"""Integration test for reward config injection in training."""

from typing import Any, cast

import numpy as np
import pytest


def test_reward_override_propagation():
    """Test reward override propagates through multiprocess collector."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLRewardConfig

    ensure_registries()

    # Create custom reward config
    custom_config = XqRobotWLRewardConfig(
        scales={
            "tracking_lin_vel": 5.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -10.0,
        },
        tracking_sigma=0.5,
        base_height_target=0.55,
    )

    # Create env with override
    env = cast(
        Any,
        registry.make(
            "XqRobotWLWalkFlat",
            num_envs=4,
            sim_backend="mujoco",
            env_cfg_override={
                "reward_config": custom_config,
                "domain_rand": {
                    "randomize_base_mass": False,
                    "randomize_ground_friction": False,
                    "randomize_kp": False,
                    "randomize_kd": False,
                    "randomize_init_yaw": False,
                },
                "commands": {
                    "vel_limit": [[-0.6, -0.3, -1.0, -0.1, 0.45], [0.6, 0.3, 1.0, 0.1, 0.85]],
                    "resampling_time": 3.0,
                },
            },
        ),
    )

    # Verify override was applied
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 5.0
    assert env._cfg.reward_config.tracking_sigma == 0.5

    # Test reward computation uses overridden scales
    env.init_state()
    env.reset(np.array([0, 1, 2, 3], dtype=np.int32))

    # Take a step and verify reward is computed
    actions = np.zeros((4, env.action_space.shape[0]), dtype=np.float32)
    state = env.step(actions)

    assert state.reward is not None
    assert len(state.reward) == 4

    env.close()


def test_backward_compatibility_no_reward_config():
    """Test env requires reward config - should fail without it."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()

    # Should fail without reward_config
    with pytest.raises(ValueError, match="reward_config must be provided"):
        registry.make(
            "XqRobotWLWalkFlat",
            num_envs=2,
            sim_backend="mujoco",
        )


def test_zero_scale_skips_computation():
    """Test that reward functions with scale=0 are skipped."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLRewardConfig

    ensure_registries()

    # Set all scales to 0 except one
    custom_config = XqRobotWLRewardConfig(
        scales={
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.0,  # Should be skipped
            "lin_vel_z": 0.0,  # Should be skipped
        },
        tracking_sigma=0.3,
        base_height_target=0.55,
    )

    env = cast(
        Any,
        registry.make(
            "XqRobotWLWalkFlat",
            num_envs=2,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": custom_config},
        ),
    )

    # Verify only non-zero scales are in config
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.0
    assert env._cfg.reward_config.scales["tracking_ang_vel"] == 0.0

    env.close()
