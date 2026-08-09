# Support Matrix

This matrix is generated conceptually from registry entries, owner YAMLs, and
tests. The generator implementation is `src/unilab/utils/support_matrix.py`; the
write target for the generated block is currently the Chinese reference page
`docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`.

## Backend Selection Rules

- The default backend is `mujoco`.
- Switch to Motrix with `--sim motrix` on the unified CLI.
- `--algo`, `--task`, and `--sim` jointly select the owner YAML.
- Do not treat `training.sim_backend` as a standalone backend switch.

## Playback Differences

- `mujoco`: `--render-mode auto` exports `play_video.mp4`.
- `motrix`: `--render-mode auto` opens an interactive renderer window; it does
  not record a video and is not bound by `play_steps`.
- `--render-mode record`: both backends record a video only.
- `--render-mode none`: no playback.

## Evidence Grades

| Grade | Repository Evidence |
| --- | --- |
| `Registered` | The env/backend pair appears after `registry.ensure_registries()`. |
| `Configured` | A matching owner YAML exists under `conf/ppo/task`, `conf/appo/task`, or `conf/offpolicy/task`. |
| `Tested` | Automated tests cover the entrypoint/task-owner/backend combination through config compose or runtime smoke. |
| `Benchmarked` | A checked-in benchmark manifest exists for the combination. |
| `Recommended` | Explicit recommendation metadata exists in the repo. |

The current generator reports no checked-in benchmark manifest and no separate
recommendation metadata, so rows do not auto-promote to `Benchmarked` or
`Recommended`.

## Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
| --- | --- | --- | --- |
| PPO (torch) | `xqrobotwl_walk_flat` | Tested | Registered |
| PPO (torch) | `xqrobotwl_walk_rough` | Tested | - |
| PPO (torch) | `xqrobotwl_jump_flat` | Tested | - |
| PPO (torch) | `xqrobotwl_jump_srl_flat` | Tested | - |
| PPO (torch) | `xqrobotwl_jump_vmc_flat` | Tested | - |
| PPO (torch) | `xqrobotwl_jump_srl_vmc_flat` | Tested | - |
| PPO (torch) | `xqrobotwl_toe_walk_flat` | Tested | - |
| PPO (torch) | `xqrobotV2_walk_flat` | Tested | Tested |
| PPO (torch) | `xqrobotV2_walk_rough` | Tested | - |
| PPO (torch) | `xqrobotV2_jump_flat` | Tested | - |
| PPO (torch) | `xqrobotV2_toe_walk_flat` | Tested | - |

## Source Index

- Registry bootstrap: `src/unilab/envs/**` registrations via
  `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/np3o/task/**`.
- Generic compose coverage:
  `tests/config/test_config_system.py::test_supported_task_composes`.
- MLX-specific compose coverage:
  `tests/config/test_config_system.py::_PPO_MLX_TASKS`.
