"""Tests for script entry-point utilities (pure functions and Hydra config defaults).

Coverage targets:
  - train_offpolicy.py: Hydra defaults, default_device(), resolve_checkpoint_path()
  - train_mlx_ppo.py:   get_latest_run(), get_latest_checkpoint()  (skipped if mlx absent)
  - play_interactive.py: resolve_checkpoint()                       (skipped if mujoco absent)
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base.backend.motrix.playback import run_motrix_playback

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

_ROOT_DIR = Path(__file__).parent.parent.parent

_CONF_DIR = Path(__file__).parent.parent.parent / "conf"

_SRC_DIR = Path(__file__).parent.parent.parent / "src"


def _normalize_overrides(overrides: list[str] | None) -> list[str]:
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        normalized.append("task=xqrobotwl_walk_flat/mujoco")
    return normalized


def _load_script(name: str) -> Any:
    """Load a scripts/<name>.py as a fresh module (no __init__ required)."""
    candidates = (
        _SCRIPTS_DIR / "training" / f"{name}.py",
        _SCRIPTS_DIR / "play" / f"{name}.py",
        _ROOT_DIR / "tools" / f"{name}.py",
        _SCRIPTS_DIR / f"{name}.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            path = candidate
            break
    else:
        raise FileNotFoundError(f"script not found: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _train_appo() -> Any:
    return _load_script("train_appo")


def test_analyze_offpolicy_trace_reports_training_e2e(tmp_path, capsys):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"name": "learner/wait_for_data", "ph": "X", "ts": 0.0, "dur": 10.0},
                    {"name": "learner/wait_for_data", "ph": "X", "ts": 1000.0, "dur": 10.0},
                    {"name": "learner/training_e2e", "ph": "X", "ts": 0.0, "dur": 2500.0},
                    {"name": "learner/weight_sync_write", "ph": "X", "ts": 800.0, "dur": 100.0},
                    {
                        "name": "learner/update_critic",
                        "ph": "X",
                        "ts": 1200.0,
                        "dur": 50.0,
                        "args": {"update_idx": 0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    mod = _load_script("analyze_offpolicy_trace")

    mod.analyze_training_e2e(trace_path)
    mod.analyze_iteration_resume_gap(trace_path)

    out = capsys.readouterr().out
    assert "training_e2e: n=1 mean=2.500ms" in out
    assert "weight_sync_end_to_next_update0_start_gap: n=1 mean=0.300ms" in out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlx_runtime_usable() -> bool:
    """Probe whether importing mlx.core is safe in a subprocess on this host."""
    if sys.platform != "darwin":
        return False
    if importlib.util.find_spec("mlx.core") is None:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import mlx.core"], capture_output=True, text=True, timeout=10
    )
    return result.returncode == 0


_HAS_MLX = _mlx_runtime_usable()

try:
    import mujoco  # noqa: F401

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False


# ---------------------------------------------------------------------------
# train_offpolicy.py — Hydra config defaults
# ---------------------------------------------------------------------------


def _ppo_cfg(overrides=None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / "ppo"), version_base="1.3"):
        return compose("config", overrides=_normalize_overrides(overrides))


def _train_rsl_rl(monkeypatch: pytest.MonkeyPatch):

    for module_name in list(sys.modules):
        if module_name == "unilab" or module_name.startswith("unilab."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    runners_mod = cast(Any, types.ModuleType("rsl_rl.runners"))
    runners_mod.OnPolicyRunner = object
    rsl_pkg = cast(Any, types.ModuleType("rsl_rl"))
    rsl_pkg.runners = runners_mod
    monkeypatch.setitem(sys.modules, "rsl_rl", rsl_pkg)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", runners_mod)
    return _load_script("train_rsl_rl")


def test_ppo_hydra_default_wandb_fields():
    cfg = _ppo_cfg()
    assert cfg.training.wandb_project == "unilab"
    assert cfg.training.wandb_entity is None
    assert cfg.training.wandb_group is None
    assert cfg.training.wandb_job_type is None
    assert cfg.training.wandb_name is None
    assert cfg.training.wandb_tags == []
    assert cfg.training.wandb_notes is None
    assert cfg.training.wandb_mode is None


def test_run_motrix_rsl_play_loop_uses_render_spacing_and_offset_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    import numpy as np
    import torch
    from tensordict import TensorDict

    mod = _train_rsl_rl(monkeypatch)

    class FakePolicy:
        def __call__(self, obs):
            batch = obs.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = []
            self.render_calls = 0

        def init_renderer(self, spacing=1.0, offset_mode="grid"):
            self.init_renderer_calls.append((spacing, offset_mode))

        def render(self):
            self.render_calls += 1

    class FakeEnv:
        def __init__(self):
            self._renderer = FakeBackend()
            self.cfg = type("Cfg", (), {"render_spacing": 2.5, "render_offset_mode": "zero"})()

        def init_play_renderer(self, render_spacing=None, render_offset_mode=None):
            offset_mode = "grid" if render_offset_mode is None else render_offset_mode
            if render_spacing is None:
                self._renderer.init_renderer(offset_mode=offset_mode)
            else:
                self._renderer.init_renderer(render_spacing, offset_mode=offset_mode)

        def render_play_frame(self):
            self._renderer.render()

        def run_playback(self, **kwargs):
            kwargs.pop("frame_state_getter", None)
            kwargs.setdefault("output_video", None)
            kwargs.setdefault("camera_kwargs", None)
            return run_motrix_playback(
                backend=self._renderer,
                env=self,
                headless=False if kwargs.get("headless") is None else bool(kwargs["headless"]),
                record_video=(
                    bool(kwargs["record_video"])
                    if kwargs.get("record_video") is not None
                    else kwargs.get("output_video") is not None
                ),
                **{k: v for k, v in kwargs.items() if k not in {"headless", "record_video"}},
            )

    class FakeWrapper:
        def __init__(self):
            self.env = FakeEnv()
            self.reset_calls = 0
            self.step_calls = 0

        def reset(self):
            self.reset_calls += 1
            return TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2), {}

        def step(self, actions):
            self.step_calls += 1
            return (
                TensorDict({"policy": torch.ones((2, 5), dtype=torch.float32)}, batch_size=2),
                torch.zeros((2,), dtype=torch.float32),
                torch.zeros((2,), dtype=torch.bool),
                {},
            )

    wrapped_env = FakeWrapper()

    mod.run_motrix_rsl_play_loop(
        wrapped_env=wrapped_env,
        policy=FakePolicy(),
        render_spacing=2.5,
        render_offset_mode="zero",
        num_steps=3,
    )

    assert wrapped_env.reset_calls == 1
    assert wrapped_env.step_calls == 3
    assert wrapped_env.env._renderer.init_renderer_calls == [(2.5, "zero")]
    assert wrapped_env.env._renderer.render_calls == 3


# ---------------------------------------------------------------------------
# train_appo.py — motrix runner / play helpers
# ---------------------------------------------------------------------------


def test_run_motrix_play_loop_runs_without_physics_state():
    import numpy as np
    import torch

    mod = _train_appo()

    class FakeActor:
        def __call__(self, td):
            batch = td.batch_size[0]
            return torch.zeros((batch, 3), dtype=torch.float32)

    class FakeBackend:
        def __init__(self):
            self.init_renderer_calls = 0
            self.render_calls = 0

        def init_renderer(self, spacing=1.0, offset_mode="grid", **kwargs):
            del spacing, offset_mode, kwargs
            self.init_renderer_calls += 1

        def render(self):
            self.render_calls += 1

    class FakeState:
        def __init__(self):
            self.obs = {"obs": np.ones((2, 5), dtype=np.float32)}

    class FakeEnv:
        def __init__(self):
            self.state = None
            self._renderer = FakeBackend()
            self.init_state_calls = 0
            self.reset_calls = 0
            self.step_calls = 0

        def init_state(self):
            self.init_state_calls += 1
            self.state = object()

        def reset(self, env_indices):
            self.reset_calls += 1
            assert env_indices.shape == (2,)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

        def step(self, actions):
            self.step_calls += 1
            assert actions.shape == (2, 3)
            return FakeState()

        def init_play_renderer(self, render_spacing=None, render_offset_mode=None):
            del render_spacing, render_offset_mode
            self._renderer.init_renderer()

        def render_play_frame(self):
            self._renderer.render()

        def run_playback(self, **kwargs):
            kwargs.pop("frame_state_getter", None)
            kwargs.setdefault("output_video", None)
            kwargs.setdefault("render_spacing", None)
            kwargs.setdefault("render_offset_mode", None)
            kwargs.setdefault("camera_kwargs", None)
            return run_motrix_playback(
                backend=self._renderer,
                env=self,
                headless=False if kwargs.get("headless") is None else bool(kwargs["headless"]),
                record_video=(
                    bool(kwargs["record_video"])
                    if kwargs.get("record_video") is not None
                    else kwargs.get("output_video") is not None
                ),
                **{k: v for k, v in kwargs.items() if k not in {"headless", "record_video"}},
            )

    env = FakeEnv()

    mod.run_motrix_play_loop(
        env=env,
        actor=FakeActor(),
        device="cpu",
        play_env_num=2,
        num_steps=3,
    )

    assert env.init_state_calls == 1
    assert env.reset_calls == 1
    assert env.step_calls == 3
    assert env._renderer.init_renderer_calls == 1
    assert env._renderer.render_calls == 3


def test_resolve_appo_checkpoint_path_prefers_latest_model_in_explicit_dir(tmp_path):
    mod = _train_appo()
    run_dir = tmp_path / "logs" / "appo" / "MyTask" / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "model_1.pt").write_bytes(b"")
    (run_dir / "model_7.pt").write_bytes(b"")

    checkpoint_path, checkpoint_dir = mod.resolve_appo_checkpoint_path(
        base_log_dir=tmp_path / "logs" / "appo" / "MyTask",
        load_run=str(run_dir),
    )

    assert checkpoint_path is not None
    assert checkpoint_path.endswith("model_7.pt")
    assert checkpoint_dir == str(run_dir)


# ---------------------------------------------------------------------------
# train_offpolicy.py — default_device()
# ---------------------------------------------------------------------------


def _offpolicy():
    return _load_script("train_offpolicy")


# ---------------------------------------------------------------------------
# train_offpolicy.py — resolve_checkpoint_path()
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_no_base_dir(tmp_path):
    """load_run='-1' with no log directory → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_explicit_existing_file(tmp_path):
    """load_run = absolute path to existing .pt → returns that path."""
    model_file = tmp_path / "model_100.pt"
    model_file.write_bytes(b"")
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", str(model_file)
    )
    assert path == str(model_file)
    assert path_dir == str(tmp_path)


def test_resolve_checkpoint_latest_picks_highest_iter(tmp_path):
    """load_run='-1' picks model with numerically highest iteration."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")
    (task_dir / "model_100.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is not None
    assert "model_100.pt" in path


def test_resolve_checkpoint_accepts_integer_latest_run(tmp_path):
    """load_run=-1 from Hydra CLI picks the latest model."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)
    (task_dir / "model_10.pt").write_bytes(b"")
    (task_dir / "model_50.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", -1)

    assert path is not None
    assert "model_50.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_explicit_run_name(tmp_path):
    """load_run = run-directory name under the log root."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "myrun"
    task_dir.mkdir(parents=True)
    (task_dir / "model_5.pt").write_bytes(b"")

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "myrun")
    assert path is not None
    assert "model_5.pt" in path
    assert path_dir == str(task_dir)


def test_resolve_checkpoint_nonexistent_explicit_path(tmp_path):
    """load_run points to a path that doesn't exist → (None, None)."""
    path, path_dir = _offpolicy().resolve_checkpoint_path(
        tmp_path, "sac", "MyTask", "/nonexistent/model.pt"
    )
    assert path is None
    assert path_dir is None


def test_resolve_checkpoint_empty_run_dir(tmp_path):
    """Run directory exists but has no model_*.pt → (None, None)."""
    task_dir = tmp_path / "logs" / "sac" / "MyTask" / "run1"
    task_dir.mkdir(parents=True)

    path, path_dir = _offpolicy().resolve_checkpoint_path(tmp_path, "sac", "MyTask", "-1")
    assert path is None


# ---------------------------------------------------------------------------
# train_mlx_ppo.py — get_latest_run() / get_latest_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path / "nonexistent") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_run(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_run_returns_last_sorted(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "2024-01-01_mujoco").mkdir()
    (tmp_path / "2024-03-15_mujoco").mkdir()
    (tmp_path / "2024-02-10_mujoco").mkdir()
    result = mod.get_latest_run(tmp_path)
    assert result is not None
    assert result.name == "2024-03-15_mujoco"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_nonexistent_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path / "no_such_dir") is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_empty_dir(tmp_path):
    mod = _load_script("train_mlx_ppo")
    assert mod.get_latest_checkpoint(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_picks_highest_iter(tmp_path):
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_0.safetensors").write_bytes(b"")
    (tmp_path / "model_50.safetensors").write_bytes(b"")
    (tmp_path / "model_200.safetensors").write_bytes(b"")
    result = mod.get_latest_checkpoint(tmp_path)
    assert result is not None
    assert result.name == "model_200.safetensors"


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_get_latest_checkpoint_ignores_non_safetensors(tmp_path):
    """Only .safetensors files count; .pt files must be ignored."""
    mod = _load_script("train_mlx_ppo")
    (tmp_path / "model_999.pt").write_bytes(b"")  # should be ignored
    assert mod.get_latest_checkpoint(tmp_path) is None


@pytest.mark.skipif(not _HAS_MLX, reason="mlx not installed")
def test_mlx_time_limit_bootstrap_values_use_final_observation():
    mod = _load_script("train_mlx_ppo")

    class FakeModel:
        def __init__(self):
            self.last_obs = None

        def value(self, obs):
            self.last_obs = obs
            return mod.mx.sum(obs, axis=1)

    state = type(
        "State",
        (),
        {
            "truncated": np.array([True, False]),
            "final_observation": {
                "obs": np.array([[3.0, 4.0], [9.0, 9.0]], dtype=np.float32),
            },
            "info": {
                "final_observation": {
                    "obs": np.array([[3.0, 4.0], [9.0, 9.0]], dtype=np.float32),
                }
            },
        },
    )()
    model = FakeModel()

    values = mod.get_time_limit_bootstrap_values(state, model, mod.mx.float32)

    if values is None:
        raise AssertionError("expected bootstrap values")
    if model.last_obs is None:
        raise AssertionError("expected model to receive observations")
    np.testing.assert_allclose(np.array(values.tolist()), np.array([7.0, 18.0], dtype=np.float32))
    np.testing.assert_allclose(
        np.array(model.last_obs.tolist()),
        np.array([[3.0, 4.0], [9.0, 9.0]], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# play_interactive.py — resolve_checkpoint()
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_nonexistent_run(tmp_path):
    """Passing a non-existent explicit path returns None."""
    mod = _load_script("play_interactive")
    result = mod.resolve_checkpoint("MyTask", str(tmp_path / "no_run"))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_dir_with_model(tmp_path):
    """Directory path containing model_*.pt files resolves to the latest."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "2024-01-01_mujoco"
    run_dir.mkdir()
    (run_dir / "model_10.pt").write_bytes(b"")
    (run_dir / "model_50.pt").write_bytes(b"")

    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is not None
    assert "model_50.pt" in result


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_explicit_file(tmp_path):
    """Absolute path to existing .pt file returns that path unchanged."""
    mod = _load_script("play_interactive")
    model_file = tmp_path / "model_99.pt"
    model_file.write_bytes(b"")
    result = mod.resolve_checkpoint("MyTask", str(model_file))
    assert result == str(model_file)


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_empty_dir(tmp_path):
    """Directory with no model_*.pt files returns None."""
    mod = _load_script("play_interactive")
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    result = mod.resolve_checkpoint("MyTask", str(run_dir))
    assert result is None


@pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")
def test_play_resolve_checkpoint_delegates_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    mod = _load_script("play_interactive")
    model_path = tmp_path / "resolved" / "model_12.pt"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"")
    captured: dict[str, object] = {}

    def _fake_resolver(root_dir, **kwargs):
        captured["root_dir"] = root_dir
        captured.update(kwargs)
        return model_path, model_path.parent

    monkeypatch.setattr(mod, "resolve_task_checkpoint_path", _fake_resolver)

    result = mod.resolve_checkpoint("MyTask", "-1", checkpoint="12", algo_log_name="custom_ppo")

    assert result == str(model_path)
    assert captured["root_dir"] == mod.ROOT_DIR
    assert captured["task_name"] == "MyTask"
    assert captured["load_run"] == "-1"
    assert captured["algo_log_name"] == "custom_ppo"
    assert captured["checkpoint"] == "12"


# ---------------------------------------------------------------------------
# play_interactive.py — RslRlVecEnvWrapper contract behavior
# ---------------------------------------------------------------------------


def _play_interactive():
    """Load play_interactive.py as a module."""
    return _load_script("play_interactive")


def test_play_wrapper_imports_shared_implementation():
    """Verify play_interactive.py uses shared RslRlVecEnvWrapper."""
    from unilab.training.rsl_rl import RslRlVecEnvWrapper as SharedWrapper

    mod = _play_interactive()
    # The wrapper class in play_interactive should be the shared one
    assert mod.RslRlVecEnvWrapper is SharedWrapper


def test_play_wrapper_uses_current_reset_contract():
    """Verify wrapper reset() uses current (obs, info) contract, not old (_, obs, _)."""
    import numpy as np
    from tensordict import TensorDict

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    # Create a fake environment that returns (obs, info) tuple
    class FakeEnv:
        def __init__(self):
            self.num_envs = 2
            self.state = type("State", (), {"obs": {"obs": np.ones((2, 5), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (3,)})()
            self.obs_groups_spec = {"obs": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            # Returns current contract: (obs, info)
            return {"obs": np.ones((2, 5), dtype=np.float32)}, {}

    env = FakeEnv()
    wrapper = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="flat")

    # Reset should work with current contract
    obs_td, info = wrapper.reset()

    assert isinstance(obs_td, TensorDict)
    assert "policy" in obs_td
    assert "actor" in obs_td
    assert obs_td.batch_size == (2,)


def test_play_wrapper_policy_obs_mode_actor():
    """Verify wrapper supports policy_obs_mode='actor'."""
    import numpy as np

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type("State", (), {"obs": {"obs": np.ones((1, 3), dtype=np.float32)}})()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {
                "obs": np.ones((1, 3), dtype=np.float32),
                "critic": np.zeros((1, 5), dtype=np.float32),
            }, {}

    env = FakeEnv()

    # Test actor mode - num_obs should match actor obs dim only
    wrapper_actor = RslRlVecEnvWrapper(env, device="cpu", policy_obs_mode="actor")
    assert wrapper_actor.num_obs == 3  # Only "obs" group
    assert wrapper_actor._actor_obs_dim == 3
    assert wrapper_actor._flat_obs_dim == 3

    obs_td, _ = wrapper_actor.reset()
    # In actor mode, policy obs should equal actor obs
    assert obs_td["policy"].shape == (1, 3)
    assert obs_td["actor"].shape == (1, 3)
    assert obs_td["critic"].shape == (1, 5)


def test_play_wrapper_flat_policy_excludes_critic_only_group():
    import numpy as np

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "critic": np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32),
                    }
                },
            )()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (7,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 4}

        def init_state(self):
            pass

        def reset(self, env_indices):
            return cast(dict[str, np.ndarray], getattr(self.state, "obs")), {}

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")
    obs_td, _ = wrapper.reset()

    np.testing.assert_allclose(
        obs_td["policy"].cpu().numpy(),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_td["critic"].cpu().numpy(),
        np.array([[9.0, 9.0, 9.0, 9.0]], dtype=np.float32),
    )
    assert wrapper.num_obs == 3
    assert wrapper.num_privileged_obs == 4


def test_play_wrapper_preserves_hora_priv_info_and_proprio_history():
    import numpy as np

    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
                        "critic": np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "critic_info": np.array([[4.0, 5.0]], dtype=np.float32),
                        "proprio_hist": np.array(
                            [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                            dtype=np.float32,
                        ),
                    },
                },
            )()
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return (
                cast(dict[str, np.ndarray], getattr(self.state, "obs")),
                cast(dict[str, np.ndarray], getattr(self.state, "info")),
            )

    wrapper = HoraRslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")
    obs_td, _ = wrapper.reset()

    np.testing.assert_allclose(
        obs_td["priv_info"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_td["proprio_hist"].cpu().numpy(),
        np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]], dtype=np.float32),
    )


def test_play_wrapper_step_exports_timeout_bootstrap_obs():
    import torch

    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (3,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 2}
            self.state = type("State", (), {"obs": {"obs": np.zeros((1, 3), dtype=np.float32)}})()

        def init_state(self):
            pass

        def reset(self, env_indices):
            return {"obs": np.zeros((1, 3), dtype=np.float32)}, {}

        def step(self, actions):
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                    "reward": np.array([1.0], dtype=np.float32),
                    "terminated": np.array([False]),
                    "truncated": np.array([True]),
                    "final_observation": {
                        "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                        "critic": np.array([[4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "final_observation": {
                            "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                            "critic": np.array([[4.0, 5.0]], dtype=np.float32),
                        }
                    },
                },
            )()

    wrapper = RslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")

    _, _, _, infos = wrapper.step(torch.zeros((1, 2)))

    assert torch.equal(infos["time_outs"], torch.tensor([True]))
    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["policy"].cpu().numpy(),
        np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["critic"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )


def test_play_wrapper_timeout_bootstrap_preserves_hora_priv_info():
    import torch

    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper

    class FakeEnv:
        def __init__(self):
            self.num_envs = 1
            self.cfg = type("Cfg", (), {"max_episode_seconds": 10.0, "ctrl_dt": 0.02})()
            self.observation_space = type("Space", (), {"shape": (5,)})()
            self.action_space = type("Space", (), {"shape": (2,)})()
            self.obs_groups_spec = {"obs": 3, "critic": 5}
            self.state = type(
                "State",
                (),
                {
                    "obs": {
                        "obs": np.zeros((1, 3), dtype=np.float32),
                        "critic": np.zeros((1, 5), dtype=np.float32),
                    },
                    "info": {
                        "critic_info": np.zeros((1, 2), dtype=np.float32),
                        "proprio_hist": np.zeros((1, 2, 3), dtype=np.float32),
                    },
                },
            )()

        def init_state(self):
            pass

        def reset(self, env_indices):
            del env_indices
            return cast(dict[str, np.ndarray], getattr(self.state, "obs")), cast(
                dict[str, np.ndarray], getattr(self.state, "info")
            )

        def step(self, actions):
            del actions
            return type(
                "StepState",
                (),
                {
                    "obs": {"obs": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)},
                    "reward": np.array([1.0], dtype=np.float32),
                    "terminated": np.array([True]),
                    "truncated": np.array([True]),
                    "final_observation": {
                        "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                        "critic": np.array([[7.0, 8.0, 9.0, 4.0, 5.0]], dtype=np.float32),
                    },
                    "info": {
                        "final_observation": {
                            "obs": np.array([[7.0, 8.0, 9.0]], dtype=np.float32),
                            "critic": np.array([[7.0, 8.0, 9.0, 4.0, 5.0]], dtype=np.float32),
                        },
                        "critic_info": np.array([[0.0, 0.0]], dtype=np.float32),
                        "proprio_hist": np.zeros((1, 2, 3), dtype=np.float32),
                    },
                },
            )()

    wrapper = HoraRslRlVecEnvWrapper(FakeEnv(), device="cpu", policy_obs_mode="flat")

    _, _, _, infos = wrapper.step(torch.zeros((1, 2)))

    np.testing.assert_allclose(
        infos["time_out_bootstrap_obs"]["priv_info"].cpu().numpy(),
        np.array([[4.0, 5.0]], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Issue #168: Unified log directory and load_run resolution
# ---------------------------------------------------------------------------


def test_ppo_hydra_default_algo_log_name():
    """Verify PPO config has algo_log_name in algo section."""
    cfg = _ppo_cfg()
    assert cfg.algo.algo_log_name == "rsl_rl_ppo"


def test_ppo_hydra_load_run_in_algo_not_training():
    """Verify load_run is in algo section, not training section (issue #168)."""
    from omegaconf import OmegaConf

    cfg = _ppo_cfg()
    assert cfg.algo.load_run == "-1"
    # training section should NOT have load_run anymore
    assert "load_run" not in cfg.training or OmegaConf.is_missing(cfg.training, "load_run")


def test_train_rsl_rl_get_log_root_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):
    """Verify _get_log_root uses algo.algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg()

    # Override algo_log_name to test
    cfg.algo.algo_log_name = "test_rsl_rl_ppo"

    log_root = mod._get_log_root(cfg)
    assert "logs/test_rsl_rl_ppo" in log_root.replace("\\", "/")


def test_train_rsl_rl_play_missing_checkpoint_skips_env_creation_and_prints_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=xqrobotwl_walk_flat/mujoco", "training.play_only=true"])
    cfg.algo.algo_log_name = "custom_ppo"

    original_root = mod.ROOT_DIR
    mod.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(
            mod,
            "create_env",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("play_rsl_rl should not create an env before checkpoint resolution")
            ),
        )

        result = mod.play_rsl_rl(cfg, device="cpu")
    finally:
        mod.ROOT_DIR = original_root

    captured = capsys.readouterr().out
    expected_task_log_root = tmp_path / "logs" / "custom_ppo" / cfg.training.task_name

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in captured
    assert "Task log root does not exist." in captured
    assert f"task_log_root={expected_task_log_root}" in captured
    assert "algo.load_run='-1'" in captured


def test_train_rsl_rl_play_reports_missing_requested_checkpoint_in_resolved_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(["task=xqrobotwl_walk_flat/mujoco", "training.play_only=true"])
    cfg.algo.algo_log_name = "custom_ppo"
    cfg.algo.checkpoint = 12

    run_dir = (
        tmp_path / "logs" / "custom_ppo" / cfg.training.task_name / "2024-01-01_00-00-00_mujoco"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_9.pt").write_bytes(b"")

    original_root = mod.ROOT_DIR
    mod.ROOT_DIR = tmp_path
    try:
        monkeypatch.setattr(
            mod,
            "create_env",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("play_rsl_rl should not create an env before checkpoint resolution")
            ),
        )

        result = mod.play_rsl_rl(cfg, device="cpu")
    finally:
        mod.ROOT_DIR = original_root

    captured = capsys.readouterr().out

    assert result is None
    assert "Could not resolve a checkpoint for play mode." in captured
    assert f"resolved_run={run_dir}" in captured
    assert "algo.checkpoint=12" in captured


def test_train_rsl_rl_motrix_auto_play_is_interactive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=xqrobotV2_walk_flat/motrix",
            "training.play_only=true",
            "training.play_steps=37",
            "training.render_spacing=2.5",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)

    class FakeEnv:
        def __init__(self):
            self.cfg = type("Cfg", (), {"render_spacing": 2.5, "render_offset_mode": "zero"})()

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "auto"
            assert kwargs["play_steps"] == 37
            plan = type(
                "Plan",
                (),
                {
                    "mode": "interactive",
                    "headless": False,
                    "record_video": False,
                    "num_steps": None,
                    "output_video": None,
                },
            )()
            kwargs["on_plan"](plan)
            captured["env"] = self
            captured.update({key: value for key, value in kwargs.items() if key != "on_plan"})
            captured["headless"] = plan.headless
            captured["record_video"] = plan.record_video
            captured["num_steps"] = plan.num_steps
            captured["output_video"] = plan.output_video
            return None

    class FakeWrapper:
        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return 0, {}

        def step(self, actions):
            return 0, 0, False, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            self.wrapped_env = wrapped_env
            self.train_cfg = train_cfg
            self.log_dir = log_dir
            self.device = device

        def load(self, path, **kwargs):
            self.loaded_path = path
            self.load_kwargs = kwargs

        def get_inference_policy(self, device):
            return lambda obs: obs

    captured: dict[str, Any] = {}

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "build_ppo_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda rl_cfg: {})
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)

    result = mod.play_rsl_rl(cfg, device="cpu")

    assert result is None
    assert captured["headless"] is False
    assert captured["record_video"] is False
    assert captured["num_steps"] is None
    assert captured["output_video"] is None
    assert captured["render_spacing"] == pytest.approx(2.5)
    assert captured["render_offset_mode"] == "zero"


def test_train_rsl_rl_record_play_uses_backend_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    mod = _train_rsl_rl(monkeypatch)
    cfg = _ppo_cfg(
        [
            "task=xqrobotV2_walk_flat/motrix",
            "training.play_only=true",
            "training.play_render_mode=record",
            "training.play_steps=37",
        ]
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "model_37.pt"
    mod.torch.save({"actor_state_dict": {}}, checkpoint)

    class FakeEnv:
        def __init__(self):
            self.cfg = type("Cfg", (), {"render_spacing": 1.0, "render_offset_mode": "grid"})()

        def run_playback_mode(self, **kwargs):
            assert kwargs["play_render_mode"] == "record"
            assert kwargs["play_steps"] == 37
            plan = type(
                "Plan",
                (),
                {
                    "mode": "record",
                    "headless": True,
                    "record_video": True,
                    "num_steps": 37,
                    "output_video": kwargs["output_video"],
                },
            )()
            kwargs["on_plan"](plan)
            captured["env"] = self
            captured.update({key: value for key, value in kwargs.items() if key != "on_plan"})
            captured["headless"] = plan.headless
            captured["record_video"] = plan.record_video
            captured["num_steps"] = plan.num_steps
            captured["output_video"] = plan.output_video
            return str(plan.output_video)

    class FakeWrapper:
        def __init__(self, env, device):
            self.env = env
            self.device = device

        def reset(self):
            return 0, {}

        def step(self, actions):
            return 0, 0, False, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            self.wrapped_env = wrapped_env
            self.train_cfg = train_cfg
            self.log_dir = log_dir
            self.device = device

        def load(self, path, map_location=None):
            self.loaded_path = path
            self.map_location = map_location

        def get_inference_policy(self, device):
            return lambda obs: obs

    captured: dict[str, Any] = {}

    monkeypatch.setattr(mod, "EXPORT_POLICY", False, raising=False)
    monkeypatch.setattr(mod, "parse_checkpoint_path", lambda *args, **kwargs: (checkpoint, run_dir))
    monkeypatch.setattr(mod, "build_ppo_play_env_cfg_override", lambda cfg: {})
    monkeypatch.setattr(mod, "create_env", lambda *args, **kwargs: FakeEnv())
    monkeypatch.setattr(mod, "_resolve_ppo_wrapper_cls", lambda rl_cfg: FakeWrapper)
    monkeypatch.setattr(mod, "normalize_ppo_train_cfg", lambda rl_cfg: {})
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)

    result = mod.play_rsl_rl(cfg, device="cpu")

    assert result == str(run_dir / "play_video.mp4")
    assert captured["headless"] is True
    assert captured["record_video"] is True
    assert captured["num_steps"] == 37
    assert captured["output_video"] == run_dir / "play_video.mp4"


def test_play_resolve_checkpoint_uses_algo_log_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Verify play_interactive.resolve_checkpoint uses algo_log_name (issue #168)."""
    monkeypatch.delenv("UNILAB_TEST_LOG_ROOT", raising=False)
    mod = _play_interactive()

    # Create test directory structure with custom algo_log_name
    run_dir = tmp_path / "logs" / "custom_ppo" / "MyTask" / "2024-01-01_mujoco"
    run_dir.mkdir(parents=True)
    (run_dir / "model_50.pt").write_bytes(b"")

    # Temporarily override ROOT_DIR to use tmp_path
    original_root = mod.ROOT_DIR
    try:
        mod.ROOT_DIR = tmp_path
        result = mod.resolve_checkpoint("MyTask", "-1", algo_log_name="custom_ppo")
        assert result is not None
        assert "model_50.pt" in result
    finally:
        mod.ROOT_DIR = original_root


def test_ppo_interactive_config_includes_playback_controls():
    cfg = _ppo_cfg()

    assert cfg.interactive.speed == pytest.approx(1.0)
    assert cfg.interactive.start_paused is False


def test_play_interactive_respects_training_device_override():
    mod = _play_interactive()
    cfg = OmegaConf.create({"training": {"device": "cpu"}})

    assert mod._select_playback_device(cfg) == "cpu"


def test_play_interactive_cli_respects_owner_action_mode_and_user_override():
    mod = _play_interactive()

    default_parsed = mod._parse_interactive_cli(
        ["--algo", "ppo", "--task", "xqrobotwl_walk_rough", "--sim", "mujoco"]
    )
    default_cfg = mod._compose_interactive_config(default_parsed.algo, default_parsed.overrides)

    assert default_cfg.interactive.action_mode == "policy"

    parsed = mod._parse_interactive_cli(
        [
            "--algo",
            "ppo",
            "--task",
            "xqrobotwl_walk_rough",
            "--sim",
            "mujoco",
            "interactive.action_mode=random",
        ]
    )
    cfg = mod._compose_interactive_config(parsed.algo, parsed.overrides)

    assert parsed.overrides == [
        "task=xqrobotwl_walk_rough/mujoco",
        "interactive.action_mode=random",
    ]
    assert cfg.interactive.action_mode == "random"


def test_play_interactive_rejects_unknown_algo_flag():
    mod = _play_interactive()

    with pytest.raises(SystemExit):
        mod._parse_interactive_cli(
            ["--algo=unknown", "--task", "xqrobotwl_walk_flat", "--sim", "mujoco"]
        )


def test_play_interactive_sac_task_shorthand_rewrites_to_owner_group():
    mod = _play_interactive()

    overrides = mod._normalize_interactive_overrides(
        "sac",
        ["task=sharpa_inhand/mujoco_hora", "algo.load_run=my_run"],
    )

    assert overrides == [
        "algo=sac",
        "task=sac/sharpa_inhand/mujoco_hora",
        "algo.load_run=my_run",
    ]


def test_play_interactive_runner_log_dir_uses_algo_log_name(monkeypatch: pytest.MonkeyPatch):

    mod = _play_interactive()
    captured: dict[str, object] = {}

    class FakeWrapper:
        def __init__(self, env, device, policy_obs_mode):
            self.env = env
            captured["policy_obs_mode"] = policy_obs_mode

        def reset(self):
            return None, {}

    class FakeRunner:
        def __init__(self, wrapped_env, train_cfg, log_dir, device):
            del wrapped_env, train_cfg, device
            captured["log_dir"] = log_dir

        def load(self, ckpt, load_cfg):
            captured["ckpt"] = ckpt
            captured["load_cfg"] = load_cfg

        def get_inference_policy(self, device):
            del device
            return object()

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_running(self):
            return False

        def sync(self):
            pass

        user_scn = type("Scene", (), {"ngeom": 0})()

    fake_env = types.SimpleNamespace(
        obs_groups_spec={"obs": 5},
        action_space=types.SimpleNamespace(shape=(3,), low=np.full((3,), -1.0), high=np.ones((3,))),
        cfg=types.SimpleNamespace(ctrl_dt=0.02),
        get_playback_model=lambda: object(),
        get_physics_state_snapshot=lambda: np.zeros((1, 8), dtype=np.float32),
    )

    monkeypatch.setattr(mod.registry, "make", lambda *args, **kwargs: fake_env)
    monkeypatch.setattr(mod, "resolve_checkpoint", lambda *args, **kwargs: "/tmp/model_10.pt")
    monkeypatch.setattr(
        mod,
        "get_entrypoint_log_root",
        lambda root_dir, *, algo_log_name, log_root=None: Path("/tmp") / algo_log_name,
    )
    monkeypatch.setattr(mod, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(mod, "OnPolicyRunner", FakeRunner)
    monkeypatch.setattr(mod, "PPOConfig", lambda: types.SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(mod.mujoco, "MjData", lambda model: object())
    monkeypatch.setattr(mod.mujoco, "mj_setState", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.mujoco, "mj_forward", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.mujoco, "mjtState", types.SimpleNamespace(mjSTATE_FULLPHYSICS=0))
    monkeypatch.setattr(mod.mujoco.viewer, "launch_passive", lambda *args, **kwargs: FakeViewer())

    args = types.SimpleNamespace(
        task="MyTask",
        load_run="-1",
        checkpoint=None,
        action_mode="policy",
        policy_obs_mode="flat",
        algo_log_name="custom_ppo",
        show_target_bodies=False,
        show_reward_debug=False,
        target_body_names="",
        target_max_bodies=0,
        target_marker_radius=0.02,
        target_axis_length=0.08,
        target_marker_alpha=0.75,
        target_show_axes=False,
        reward_debug_show_velocity=False,
        reward_debug_lin_vel_scale=0.08,
        reward_debug_ang_vel_scale=0.05,
        reward_debug_show_connectors=False,
        reward_debug_show_global_anchor=False,
        speed=1.0,
        start_paused=False,
    )

    mod.play_interactive(args)

    assert captured["ckpt"] == "/tmp/model_10.pt"
    assert captured["log_dir"].replace("\\", "/") == "/tmp/custom_ppo/MyTask/play_temp"


def test_play_interactive_import_does_not_swallow_registry_bootstrap_errors(
    monkeypatch: pytest.MonkeyPatch,
):

    play_interactive_path = _SCRIPTS_DIR / "play" / "play_interactive.py"
    training_mod = cast(Any, types.ModuleType("unilab.training"))

    def _fail_bootstrap() -> None:
        raise RuntimeError("bootstrap failed")

    training_mod.ensure_registries = _fail_bootstrap
    training_mod.get_entrypoint_log_root = lambda *args, **kwargs: Path("/tmp")
    training_mod.resolve_task_checkpoint_path = lambda *args, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "unilab.training", training_mod)

    mujoco_mod = cast(Any, types.ModuleType("mujoco"))
    mujoco_mod.viewer = cast(Any, types.ModuleType("mujoco.viewer"))
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", mujoco_mod.viewer)

    spec = importlib.util.spec_from_file_location(
        "play_interactive_test_module", play_interactive_path
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
