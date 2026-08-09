# 运动控制

运动控制任务注册在 `src/unilab/envs/locomotion/` 中。`conf/` 下可用的
owner YAML 定义了哪些算法与后端组合是可运行的。

## 系列

- XqRobotWL：`xqrobotwl_walk_flat`、`xqrobotwl_walk_rough`、`xqrobotwl_jump_flat`、
  `xqrobotwl_jump_srl_flat`、`xqrobotwl_jump_vmc_flat`、`xqrobotwl_jump_srl_vmc_flat`、
  `xqrobotwl_toe_walk_flat`
- XqRobotV2：`xqrobotV2_walk_flat`、`xqrobotV2_walk_rough`、`xqrobotV2_jump_flat`、
  `xqrobotV2_toe_walk_flat`

## 示例

```bash
uv run train --algo ppo --task xqrobotwl_walk_flat --sim mujoco
uv run train --algo ppo --task xqrobotwl_walk_rough --sim mujoco training.no_play=true
uv run train --algo ppo --task xqrobotwl_jump_flat --sim mujoco training.no_play=true
uv run train --algo ppo --task xqrobotV2_walk_flat --sim motrix training.no_play=true
```

查看支持矩阵以了解按 entrypoint、task owner 和 backend 划分的证据分级：
{doc}`../../5-reference/5-support_matrix`。
