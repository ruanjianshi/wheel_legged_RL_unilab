# Locomotion

Locomotion tasks are registered in `src/unilab/envs/locomotion/`. The available
owner YAMLs under `conf/` define which algorithm and backend combinations are
runnable.

## Families

- XqRobotWL: `xqrobotwl_walk_flat`, `xqrobotwl_walk_rough`, `xqrobotwl_jump_flat`,
  `xqrobotwl_jump_srl_flat`, `xqrobotwl_jump_vmc_flat`, `xqrobotwl_jump_srl_vmc_flat`,
  `xqrobotwl_toe_walk_flat`
- XqRobotV2: `xqrobotV2_walk_flat`, `xqrobotV2_walk_rough`, `xqrobotV2_jump_flat`,
  `xqrobotV2_toe_walk_flat`

## Examples

```bash
uv run train --algo ppo --task xqrobotwl_walk_flat --sim mujoco
uv run train --algo ppo --task xqrobotwl_walk_rough --sim mujoco training.no_play=true
uv run train --algo ppo --task xqrobotwl_jump_flat --sim mujoco training.no_play=true
uv run train --algo ppo --task xqrobotV2_walk_flat --sim motrix training.no_play=true
```

Check the support matrix for evidence grade by entrypoint, task owner, and
backend: {doc}`../../5-reference/5-support_matrix`.
