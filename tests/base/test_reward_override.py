"""Test reward config override through registry."""

from typing import Any, cast

import pytest

from unilab.base import registry
from unilab.base.registry import ensure_registries


def test_reward_override_xqrobotwl():
    """Test XqRobotWL reward config override."""
    ensure_registries()

    from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLRewardConfig

    override_config = XqRobotWLRewardConfig(
        scales={"tracking_lin_vel": 999.0},
        tracking_sigma=0.5,
        base_height_target=0.5,
    )

    env = cast(
        Any,
        registry.make(
            "XqRobotWLWalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": override_config},
        ),
    )

    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 999.0
    env.close()


def test_reward_override_xqrobotV2():
    """Test XqRobotV2 reward config override."""
    ensure_registries()

    from unilab.envs.locomotion.xqrobotV2.joystick import XqRobotRewardConfig

    override_config = XqRobotRewardConfig(
        scales={"tracking_lin_vel": 888.0, "alive": 20.0},
        tracking_sigma=0.3,
        base_height_target=0.65,
        min_base_height=0.2,
        max_tilt_deg=60.0,
    )

    env = cast(
        Any,
        registry.make(
            "XqRobotV2WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": override_config},
        ),
    )

    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 888.0
    assert env._cfg.reward_config.scales["alive"] == 20.0
    env.close()
