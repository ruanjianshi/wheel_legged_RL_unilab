"""Tests for structured configs and Hydra YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

CONF_DIR = Path(__file__).parent.parent.parent / "conf"


# ---------------------------------------------------------------------------
# structured_configs dataclass defaults
# ---------------------------------------------------------------------------


def test_sac_config_defaults():
    from unilab.structured_configs import SACAlgoParams, SACConfig

    cfg = SACConfig()
    assert cfg.algo == "sac"
    assert cfg.num_envs == 4096
    assert cfg.batch_size == 8192
    assert cfg.use_symmetry is False
    assert isinstance(cfg.algo_params, SACAlgoParams)
    assert cfg.algo_params.alpha_init == 0.01
    assert cfg.algo_params.use_compile is True


def test_td3_config_defaults():
    from unilab.structured_configs import TD3Config

    cfg = TD3Config()
    assert cfg.algo == "td3"
    assert cfg.num_envs == 4096
    assert cfg.use_layer_norm is False
    assert cfg.algo_params.weight_decay == 0.1


def test_flashsac_config_defaults():
    from unilab.structured_configs import FlashSACAlgoParams, FlashSACConfig

    cfg = FlashSACConfig()
    assert cfg.algo == "flashsac"
    assert cfg.num_envs == 1024
    assert cfg.batch_size == 2048
    assert cfg.learning_starts == 98
    assert cfg.gamma == pytest.approx(0.97)
    assert cfg.obs_normalization is False
    assert isinstance(cfg.algo_params, FlashSACAlgoParams)
    assert cfg.algo_params.normalize_reward is True
    assert cfg.algo_params.amp_dtype == "auto"
    assert cfg.algo_params.use_compile is True


def test_ppo_config_defaults():
    from unilab.structured_configs import PPOConfig

    cfg = PPOConfig()
    assert cfg.algo == "ppo"
    assert cfg.max_iterations == 101
    assert cfg.algorithm.clip_param == 0.2
    assert cfg.algorithm.class_name == "unilab.algos.torch.rsl_rl_ppo:FinalObservationAwarePPO"
    assert cfg.algorithm.enable_compile is True
    assert cfg.policy.class_name == "ActorCritic"


def test_appo_config_defaults():
    from unilab.structured_configs import APPOConfig

    cfg = APPOConfig()
    assert cfg.algo == "appo"
    assert cfg.num_envs == 2048
    assert cfg.actor.class_name == "rsl_rl.models.MLPModel"
    assert cfg.algorithm.enable_compile is True


def test_base_config_to_dict():
    from unilab.structured_configs import SACConfig

    cfg = SACConfig()
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["algo"] == "sac"
    assert "algo_params" in d
    assert isinstance(d["algo_params"], dict)


# ---------------------------------------------------------------------------
# Hydra YAML loading — ppo
# ---------------------------------------------------------------------------


def test_ppo_compile_overrides():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose(
            "config",
            overrides=[
                "task=xqrobotwl_walk_flat/mujoco",
                "algo.algorithm.enable_compile=false",
            ],
        )
    assert cfg.algo.algorithm.enable_compile is False


def test_ppo_xqrobotwl_walk_flat_mujoco_composes():
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / "ppo"), version_base="1.3"):
        cfg = compose("config", overrides=["task=xqrobotwl_walk_flat/mujoco"])
    assert cfg.training.task_name == "XqRobotWLWalkFlat"
    assert cfg.training.sim_backend == "mujoco"


# ---------------------------------------------------------------------------
# Issue #197 DoD: rough terrain profile params overridable via Hydra
# ---------------------------------------------------------------------------


def test_apply_cfg_overrides_deep_merges_dataclass_field():
    """registry.apply_cfg_overrides must deep-merge into existing dataclass
    instances rather than re-instantiating them, so partial overrides like
    `scene.terrain.generator.num_rows=4` keep `sub_terrains` and other defaults."""
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs.locomotion.xqrobotwl.rough import XqRobotWLWalkRoughCfg

    cfg = XqRobotWLWalkRoughCfg()
    cfg.scene.terrain.generator.num_cols = 3
    cfg.scene.terrain.generator.border_width = 2.5
    cfg.scene.terrain.generator.sub_terrains = {
        "test_flat": cfg.scene.terrain.generator.sub_terrains["flat"]
    }
    cfg.scene.terrain.generator.add_lights = False
    apply_cfg_overrides(
        cfg,
        {"scene": {"terrain": {"generator": {"num_rows": 4, "seed": 42, "curriculum": True}}}},
    )

    # Overridden fields take effect.
    assert cfg.scene.terrain.generator.num_rows == 4
    assert cfg.scene.terrain.generator.seed == 42
    assert cfg.scene.terrain.generator.curriculum is True
    # Non-overridden fields preserve the pre-existing instance state.
    assert cfg.scene.terrain.generator.num_cols == 3
    assert cfg.scene.terrain.generator.border_width == pytest.approx(2.5)
    assert list(cfg.scene.terrain.generator.sub_terrains) == ["test_flat"]
    assert cfg.scene.terrain.generator.add_lights is False
