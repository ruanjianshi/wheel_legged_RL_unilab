"""Test Hydra config composition across algos and task owners."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

CONF_DIR = Path(__file__).parent.parent.parent / "conf"
_PPO_MLX_TASKS = {"xqrobotwl_walk_flat", "xqrobotV2_walk_flat"}
_BACKENDS = ("mujoco", "motrix")


def _expected_backend_from_variant(name: str) -> str | None:
    for backend in _BACKENDS:
        if name == backend or name.startswith(f"{backend}_"):
            return backend
    return None


def _compose(algo_dir: str, config_name: str = "config", overrides: list[str] | None = None):
    normalized_overrides = _normalize_overrides(algo_dir, overrides)

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR / algo_dir), version_base="1.3"):
        return compose(config_name, overrides=normalized_overrides)


def _normalize_overrides(algo_dir: str, overrides: list[str] | None) -> list[str]:
    normalized: list[str] = []
    task_selected = False

    for override in overrides or []:
        if override.startswith("algo="):
            normalized.append(override)
            continue
        if override.startswith("task="):
            task_selected = True
            normalized.append(override)
            continue
        normalized.append(override)

    if not task_selected:
        normalized.append("task=xqrobotwl_walk_flat/mujoco")

    return normalized


def _assert_reward_populated(cfg, label: str):
    assert hasattr(cfg, "reward"), f"{label} missing cfg.reward"
    reward_dict = OmegaConf.to_container(cfg.reward, resolve=True)
    assert isinstance(reward_dict, dict), f"{label} reward must resolve to mapping"
    assert "scales" in reward_dict, f"{label} reward must contain scales"
    assert len(reward_dict["scales"]) > 0, f"{label} reward.scales must be non-empty"


def _supported_task_cases() -> list[tuple[str, str, str, str, str, list[str]]]:
    cases: list[tuple[str, str, str, str, str, list[str]]] = []

    for algo_dir in ["ppo", "appo"]:
        root = CONF_DIR / algo_dir / "task"
        if not root.exists():
            continue
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for backend_file in sorted(task_dir.glob("*.yaml")):
                expected_backend = _expected_backend_from_variant(backend_file.stem)
                if expected_backend is None:
                    continue
                cases.append(
                    (
                        algo_dir,
                        "config",
                        task_dir.name,
                        expected_backend,
                        str(backend_file.relative_to(CONF_DIR)),
                        [f"task={task_dir.name}/{backend_file.stem}"],
                    )
                )
                if algo_dir == "ppo" and task_dir.name in _PPO_MLX_TASKS:
                    cases.append(
                        (
                            algo_dir,
                            "config_mlx",
                            task_dir.name,
                            expected_backend,
                            str(backend_file.relative_to(CONF_DIR)),
                            [f"task={task_dir.name}/{backend_file.stem}"],
                        )
                    )

    offpolicy_root = CONF_DIR / "offpolicy" / "task"
    if offpolicy_root.exists():
        for algo_root in sorted(path for path in offpolicy_root.iterdir() if path.is_dir()):
            for task_dir in sorted(path for path in algo_root.iterdir() if path.is_dir()):
                for backend_file in sorted(task_dir.glob("*.yaml")):
                    expected_backend = _expected_backend_from_variant(backend_file.stem)
                    if expected_backend is None:
                        continue
                    cases.append(
                        (
                            "offpolicy",
                            "config",
                            task_dir.name,
                            expected_backend,
                            str(backend_file.relative_to(CONF_DIR)),
                            [
                                f"algo={algo_root.name}",
                                f"task={algo_root.name}/{task_dir.name}/{backend_file.stem}",
                            ],
                        )
                    )

    return cases


@pytest.mark.parametrize(
    "algo_dir,config_name",
    [
        ("ppo", "config"),
        ("ppo", "config_mlx"),
    ],
)
def test_algo_config_composes(algo_dir: str, config_name: str):
    cfg = _compose(algo_dir, config_name)
    assert cfg.training.task_name
    assert cfg.training.sim_backend == "mujoco"


def test_legacy_config_groups_removed():
    for path in [
        CONF_DIR / "ppo" / "reward",
        CONF_DIR / "ppo" / "backend_task_preset",
        CONF_DIR / "ppo" / "algo_preset",
        CONF_DIR / "ppo" / "sim_backend",
        CONF_DIR / "appo" / "reward",
        CONF_DIR / "appo" / "backend_task_preset",
        CONF_DIR / "appo" / "sim_backend",
        CONF_DIR / "offpolicy" / "reward",
        CONF_DIR / "offpolicy" / "backend_task_preset",
        CONF_DIR / "offpolicy" / "algo_preset",
        CONF_DIR / "offpolicy" / "sim_backend",
    ]:
        assert not path.exists(), f"legacy config group should be removed: {path}"


def test_task_files_keep_full_identity_without_hidden_backend_marker():
    for path in sorted(CONF_DIR.glob("*/task/**/*.yaml")):
        cfg = OmegaConf.load(path)
        cfg_dict_raw = OmegaConf.to_container(cfg, resolve=True) or {}
        assert isinstance(cfg_dict_raw, dict)
        assert "_selected_sim_backend" not in cfg_dict_raw, (
            f"task has hidden backend marker: {path}"
        )
        if path.stem not in _BACKENDS:
            continue
        training_raw = cfg_dict_raw.get("training", {})
        assert isinstance(training_raw, dict)
        assert "task_name" in training_raw, f"task missing task_name: {path}"
        assert "sim_backend" in training_raw, f"task missing sim_backend: {path}"


def test_motrix_task_files_do_not_declare_post_step_forward_sensor():
    for path in sorted(CONF_DIR.glob("*/task/**/*motrix*.yaml")):
        cfg = OmegaConf.load(path)

        assert OmegaConf.select(cfg, "env.post_step_forward_sensor") is None, (
            "post_step_forward_sensor is routed only to MuJoCo backends: "
            f"{path.relative_to(CONF_DIR)}"
        )


@pytest.mark.parametrize(
    "algo_dir,config_name,task,backend,task_file,overrides",
    _supported_task_cases(),
)
def test_supported_task_composes(
    algo_dir: str,
    config_name: str,
    task: str,
    backend: str,
    task_file: str,
    overrides: list[str],
):
    cfg = _compose(algo_dir, config_name, overrides=overrides)

    assert cfg.training.task_name, f"{task_file} should resolve task_name"
    assert cfg.training.sim_backend == backend, f"{task_file} should set backend"
    _assert_reward_populated(cfg, task_file)


def test_cli_override_beats_task_defaults():
    cfg = _compose(
        "ppo",
        overrides=["task=xqrobotV2_walk_flat/motrix", "algo.max_iterations=1"],
    )

    assert cfg.algo.max_iterations == 1
    assert cfg.training.sim_backend == "motrix"
