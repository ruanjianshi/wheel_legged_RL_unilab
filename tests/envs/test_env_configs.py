"""Tests for env config completeness and env instantiation.

Config-attribute tests (non-slow) verify that config dataclasses expose every
attribute accessed by their paired env class, WITHOUT running a simulation.

Slow tests actually call registry.make() and run reset + step.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from unilab.base.registry import ensure_registries


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco", reason="mujoco not installed")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env not available (platform/libstdc++ issue)")


# ---------------------------------------------------------------------------
# Non-slow: config attribute completeness (no env.step(), no MuJoCo sim)
# ---------------------------------------------------------------------------


def test_registry_bootstrap_and_config_imports_do_not_require_mujoco():
    repo_root = Path(__file__).parents[2]
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "mujoco" or name.startswith("mujoco."):
                raise ImportError("mujoco blocked by test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        from unilab.base import registry
        from unilab.base.backend import create_backend
        from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLWalkFlatCfg
        from unilab.envs.locomotion.xqrobotV2.joystick import XqRobotV2WalkFlatCfg
        from unilab.base.registry import ensure_registries

        ensure_registries()
        assert callable(create_backend)
        assert registry.contains("XqRobotWLWalkFlat")
        assert registry.contains("XqRobotV2WalkFlat")
        XqRobotWLWalkFlatCfg()
        XqRobotV2WalkFlatCfg()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout


# ---------------------------------------------------------------------------
# Fast env/backend smoke tests
# ---------------------------------------------------------------------------

# Environments that don't need special config overrides
_STANDARD_ENVS = [
    "XqRobotWLWalkFlat",
    "XqRobotWLWalkRough",
    "XqRobotV2WalkFlat",
    "XqRobotV2WalkRough",
]


@pytest.mark.parametrize("env_name", _STANDARD_ENVS)
def test_env_reset_and_step(env_name: str, default_xq_reward_config):
    """Every registered env must be constructible, resetable, and steppable.

    Verifies:
    - observation/action spaces are valid
    - init_state + reset produces dict obs with correct keys and shapes
    - step with zero actions produces dict obs, scalar reward, bool done
    """
    _require_mujoco_runtime()
    ensure_registries()
    from unilab.base import registry

    # Provide reward_config for envs that require it via Hydra. Disable DR knobs
    # that need cached per-robot tables (mirrors the walk task YAMLs).
    env_cfg_override: dict[str, Any] = {
        "reward_config": default_xq_reward_config,
        "domain_rand": {
            "randomize_base_mass": False,
            "randomize_ground_friction": False,
            "randomize_kp": False,
            "randomize_kd": False,
            "randomize_init_yaw": False,
        },
    }
    # Flat envs use 5D commands [vx, vy, vyaw, tsk, height] (see task YAML);
    # rough envs default to 4D via their own Commands subclass.
    if "WalkFlat" in env_name:
        env_cfg_override["commands"] = {
            "vel_limit": [[-0.6, -0.3, -1.0, -0.1, 0.45], [0.6, 0.3, 1.0, 0.1, 0.85]],
            "resampling_time": 3.0,
        }

    env = cast(
        Any,
        registry.make(
            env_name, num_envs=2, sim_backend="mujoco", env_cfg_override=env_cfg_override
        ),
    )
    try:
        # 1. Spaces
        obs_space = env.observation_space
        act_space = env.action_space
        assert obs_space.shape is not None and obs_space.shape[0] > 0
        assert act_space.shape is not None and act_space.shape[0] > 0

        # obs_groups_spec must sum to observation_space total dim
        spec = env.obs_groups_spec
        assert isinstance(spec, dict)
        assert sum(spec.values()) == obs_space.shape[0]

        # 2. Reset
        state = env.init_state()
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert key in state.obs, f"obs missing group '{key}'"
            assert state.obs[key].shape == (2, dim), (
                f"obs['{key}'] shape mismatch: {state.obs[key].shape} != (2, {dim})"
            )

        # 3. Step with zero actions
        actions = np.zeros((2, act_space.shape[0]))
        state = env.step(actions)
        assert isinstance(state.obs, dict)
        for key, dim in spec.items():
            assert state.obs[key].shape == (2, dim)
        assert state.reward.shape == (2,)
        assert state.terminated.shape == (2,)
        assert state.truncated.shape == (2,)
    finally:
        env.close()
