"""Test reward config injection system."""

from typing import Any, cast

import pytest
from hydra import compose, initialize


def test_resolve_reward_dict_reads_task_reward():
    """Task-backend configs should expose the final reward mapping directly."""
    from unilab.training.reward import resolve_reward_dict

    with initialize(config_path="../../conf/ppo", version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["task=xqrobotV2_walk_flat/mujoco"],
        )

    reward_dict = resolve_reward_dict(cfg)

    assert reward_dict["scales"]["tracking_lin_vel"] == 1.5
    assert reward_dict["scales"]["tracking_ang_vel"] == 1.5


def test_reward_config_conversion():
    """Test reward config converts to dataclasses via registry."""
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()

    # Test XqRobotWL walk config - registry auto-converts dict to XqRobotWLRewardConfig
    xqwl_dict = {
        "scales": {"tracking_lin_vel": 1.5, "alive": 1.0},
        "tracking_sigma": 0.3,
        "base_height_target": 0.55,
        "min_base_height": 0.2,
        "max_tilt_deg": 60.0,
    }
    env = cast(
        Any,
        registry.make(
            "XqRobotWLWalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": xqwl_dict},
        ),
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.5
    env.close()

    # Test XqRobotV2 config - registry auto-converts dict to XqRobotRewardConfig
    xq2_dict = {
        "scales": {"tracking_lin_vel": 1.5, "base_height": -5.0},
        "tracking_sigma": 0.3,
        "base_height_target": 0.65,
    }
    env = cast(
        Any,
        registry.make(
            "XqRobotV2WalkFlat",
            num_envs=1,
            sim_backend="mujoco",
            env_cfg_override={"reward_config": xq2_dict},
        ),
    )
    assert hasattr(env._cfg.reward_config, "scales")
    assert env._cfg.reward_config.scales["tracking_lin_vel"] == 1.5
    env.close()
