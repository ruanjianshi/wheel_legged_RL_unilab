from __future__ import annotations

from dataclasses import dataclass

from unilab.envs.locomotion.xqrobotV2.joystick import XqRobotV2WalkFlatCfg
from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLWalkFlatCfg
from unilab.envs.locomotion.xqrobotwl.rough import XqRobotWLWalkRoughCfg


@dataclass(frozen=True)
class LocomotionTaskSpec:
    owner_task_id: str
    env_task_name: str
    display_name: str
    config_cls: type


_TASK_SPECS = {
    "xqrobotwl_walk_flat": LocomotionTaskSpec(
        owner_task_id="xqrobotwl_walk_flat",
        env_task_name="XqRobotWLWalkFlat",
        display_name="xqrobotwl_walk_flat",
        config_cls=XqRobotWLWalkFlatCfg,
    ),
    "xqrobotwl_walk_rough": LocomotionTaskSpec(
        owner_task_id="xqrobotwl_walk_rough",
        env_task_name="XqRobotWLWalkRough",
        display_name="xqrobotwl_walk_rough",
        config_cls=XqRobotWLWalkRoughCfg,
    ),
    "xqrobotV2_walk_flat": LocomotionTaskSpec(
        owner_task_id="xqrobotV2_walk_flat",
        env_task_name="XqRobotV2WalkFlat",
        display_name="xqrobotV2_walk_flat",
        config_cls=XqRobotV2WalkFlatCfg,
    ),
}
_TASK_ALIASES = {spec.env_task_name: spec.owner_task_id for spec in _TASK_SPECS.values()}
_TASK_ALIASES.update({f"task={task_id}/mujoco": task_id for task_id in _TASK_SPECS})
_TASK_ALIASES.update({f"{task_id}/mujoco": task_id for task_id in _TASK_SPECS})


def canonical_locomotion_task_ids() -> list[str]:
    return list(_TASK_SPECS.keys())


def normalize_locomotion_task_id(task_name: str) -> str:
    normalized = task_name.strip()
    if normalized.startswith("task="):
        normalized = normalized[len("task=") :]
    if normalized.endswith("/motrix"):
        raise ValueError(
            f"Task '{task_name}' targets motrix, but this benchmark only measures MuJoCo paths."
        )
    if normalized in _TASK_SPECS:
        return normalized
    alias_target = _TASK_ALIASES.get(normalized)
    if alias_target is not None:
        return alias_target
    raise ValueError(
        f"Unknown task '{task_name}'. Available task ids: {canonical_locomotion_task_ids()}. "
        "Accepted aliases also include the legacy env names and task=<name>/mujoco forms."
    )


def locomotion_task_spec(task_name: str) -> LocomotionTaskSpec:
    return _TASK_SPECS[normalize_locomotion_task_id(task_name)]


def locomotion_task_model_file(task_name: str) -> str:
    cfg = locomotion_task_spec(task_name).config_cls()
    scene = getattr(cfg, "scene", None)
    model_file = getattr(scene, "model_file", None)
    if model_file:
        return str(model_file)

    raise ValueError(f"{type(cfg).__name__} does not define scene.model_file")


def locomotion_env_name(task_name: str) -> str:
    return locomotion_task_spec(task_name).env_task_name
