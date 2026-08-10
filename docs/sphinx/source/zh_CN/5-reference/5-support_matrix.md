# 后端支持矩阵

本页是后端参考页，放生成矩阵和需要精确查证的 backend 规则。它不承担首次阅读职责。

## 适合谁看

- 想按 task owner / algorithm / backend 精确查支持状态
- 想知道 `Registered`、`Configured`、`Tested` 的证据差异
- 想确认 playback 和 owner compose 的 backend 规则

## Backend 选择规则

- 默认后端是 `mujoco`
- 切到 Motrix 用统一 CLI 的 `--sim motrix`
- `--algo`、`--task`、`--sim` 共同选择 owner YAML
- 不要把 `training.sim_backend` 当独立 backend switch

## Playback Differences

- `mujoco`: `--render-mode auto` 会导出 `play_video.mp4`
- `motrix`: `--render-mode auto` 会打开交互式 renderer 窗口，不录制视频，不受 `play_steps` 限制
- `--render-mode record`: 两个后端都只录制视频
- `--render-mode none`: 不回放

## Support Matrix

下面的矩阵由 registry、owner YAML 和测试清单自动汇总；不要手工编辑表格内容。需要刷新时运行：

```bash
uv run scripts/generate_support_matrix.py --write
```

<!-- BEGIN GENERATED SUPPORT MATRIX -->
### Evidence Grades

| 等级 | 仓库事实来源 |
|------|--------------|
| `Registered` | `ensure_registries()` 导入后的 `registry.list_registered_envs()` 中存在该 env/backend。 |
| `Configured` | 存在对应的 owner YAML：`conf/{ppo,appo,offpolicy}/task/...`。 |
| `Tested` | `tests/` 中有自动化覆盖该 entrypoint/task owner/backend 组合。这里的 `Tested` 包含 config compose 与脚本/运行时测试，不等同于默认推荐路径。 |
| `Benchmarked` | 存在与该组合绑定的已提交 benchmark manifest。 |
| `Recommended` | 仓库中存在显式 recommendation 元数据。 |

`Tested` 只描述仓库中已有自动化覆盖，不代表该组合具备同名 MuJoCo owner 的全部 backend capability；例如 phase-1 Motrix owner 可能只覆盖训练 smoke 和明确启用的 DR 子集。

未检测到与这些组合绑定的已提交 benchmark manifest，因此当前不会自动提升到 `Benchmarked`。
仓库中目前也没有单独的 recommendation 元数据，因此当前不会自动提升到 `Recommended`。

### Entrypoint x Task Owner

| Entrypoint | Task owner | MuJoCo | Motrix |
|------------|------------|--------|--------|
| PPO (torch) | `xqrobotwl_walk_flat` (XqRobotWL walk flat) | Tested | Registered |
| PPO (torch) | `xqrobotwl_walk_rough` (XqRobotWL walk rough) | Tested | - |
| PPO (torch) | `xqrobotwl_jump_flat` (XqRobotWL jump flat) | Tested | - |
| PPO (torch) | `xqrobotwl_jump_srl_flat` (XqRobotWL jump SRL flat) | Tested | - |
| PPO (torch) | `xqrobotwl_jump_vmc_flat` (XqRobotWL jump VMC flat) | Tested | - |
| PPO (torch) | `xqrobotwl_jump_srl_vmc_flat` (XqRobotWL jump SRL+VMC flat) | Tested | - |
| PPO (torch) | `xqrobotwl_toe_walk_flat` (XqRobotWL toe-walk flat) | Tested | - |
| PPO (torch) | `xqrobotV2_walk_flat` (XqRobotV2 walk flat) | Tested | Tested |
| PPO (torch) | `xqrobotV2_walk_rough` (XqRobotV2 walk rough) | Tested | - |
| PPO (torch) | `xqrobotV2_jump_flat` (XqRobotV2 jump flat) | Tested | - |
| PPO (torch) | `xqrobotV2_toe_walk_flat` (XqRobotV2 toe-walk flat) | Tested | - |
| PPO (torch) | `xqrobotwl_backflip_flat` (xqrobotwl backflip flat) | Tested | - |
| PPO (torch) | `xqrobotwl_single_leg_flat` (xqrobotwl single leg flat) | Tested | - |
| PPO (torch) | `xqrobotwl_single_leg_move` (xqrobotwl single leg move) | Tested | - |
| PPO (torch) | `xqrobotwl_single_leg_unicycle` (xqrobotwl single leg unicycle) | Tested | - |
| PPO (mlx) | `xqrobotwl_walk_flat` (XqRobotWL walk flat) | Tested | Registered |
| PPO (mlx) | `xqrobotwl_walk_rough` (XqRobotWL walk rough) | Configured | - |
| PPO (mlx) | `xqrobotwl_jump_flat` (XqRobotWL jump flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_jump_srl_flat` (XqRobotWL jump SRL flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_jump_vmc_flat` (XqRobotWL jump VMC flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_jump_srl_vmc_flat` (XqRobotWL jump SRL+VMC flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_toe_walk_flat` (XqRobotWL toe-walk flat) | Configured | - |
| PPO (mlx) | `xqrobotV2_walk_flat` (XqRobotV2 walk flat) | Tested | Tested |
| PPO (mlx) | `xqrobotV2_walk_rough` (XqRobotV2 walk rough) | Configured | - |
| PPO (mlx) | `xqrobotV2_jump_flat` (XqRobotV2 jump flat) | Configured | - |
| PPO (mlx) | `xqrobotV2_toe_walk_flat` (XqRobotV2 toe-walk flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_backflip_flat` (xqrobotwl backflip flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_single_leg_flat` (xqrobotwl single leg flat) | Configured | - |
| PPO (mlx) | `xqrobotwl_single_leg_move` (xqrobotwl single leg move) | Configured | - |
| PPO (mlx) | `xqrobotwl_single_leg_unicycle` (xqrobotwl single leg unicycle) | Configured | - |

### Source Index

- Registry bootstrap: `src/unilab/envs/**` decorators via `unilab.base.registry.ensure_registries()`.
- Owner YAML scan: `conf/ppo/task/**`, `conf/np3o/task/**`.
- Generic compose coverage: `tests/config/test_config_system.py::test_supported_task_composes`.
<!-- END GENERATED SUPPORT MATRIX -->
