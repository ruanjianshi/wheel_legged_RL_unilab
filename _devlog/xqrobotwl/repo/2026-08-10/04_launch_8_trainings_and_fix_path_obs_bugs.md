# [35] 启动 8 个 xqrobotwl 训练 + 修复 shell 路径与 stairs obs 维度 bug

**日期**: 2026-08-10
**来源**: 用户要求启动 shell/xqrobotwl 下全部训练 (8 个脚本)
**关联**: [[32_shell_reorganize_by_task]], [[34_add_jump_vmc_eval_scripts]]

---

## 问题描述

启动训练时发现两个 bug:

1. **shell 脚本 ROOT_DIR 路径错误**: shell 细分到 `shell/<robot>/<task>/` 后,
   脚本内 `ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"` 从新层级上溯只到
   `shell/`, 导致 `scripts/training/train_rsl_rl.py` 找不到 → 训练无法启动。

2. **stairs env obs 维度 bug**: `XqRobotWLStairsEnv` 继承 rough env
   (`_obs_frame_dim=32`, 4D commands), 但 stairs 任务 YAML 用 5D commands
   (带 height) → `_compute_obs` 拼出 33 维 obs_frame 写入 32 维 buffer →
   `ValueError: could not broadcast (1024,33) into (1024,32)` → stairs 崩溃。

## 根因分析

1. shell 细分时引入: `shell/<robot>` → `shell/<robot>/<task>` 深了一层,
   但 `ROOT_DIR` 的 `../..` 未同步加深。
2. `_obs_frame_dim=32` 是 07-25 (38bcd16) 为 rough 的 4D commands 引入的,
   stairs 一直用 5D commands 继承 rough → 07-25 后 stairs 训练从未成功
   (最后成功 checkpoint 是 07-21, actor 输入 297 = 33×9)。

## 解决方案

1. 批量修复所有 shell 脚本 `dirname "$0")/../..` → `dirname "$0")/../../..`
   (从 `shell/<robot>/<task>` 上溯 3 层到仓库根)。
2. `XqRobotWLStairsEnv.__init__` 覆盖 `_obs_frame_dim=33` / `_critic_frame_dim=36`
   并重建 history buffers (与 flat env 的 5D commands 一致), 加 `import numpy as np`。

## 修改文件

- `shell/**/*.sh` (30 个): ROOT_DIR 层级修正
- `src/unilab/envs/locomotion/xqrobotwl/stairs.py`: 覆盖 obs/critic 维度

## 验证

- `bash -n` 全部 shell 脚本通过; ROOT_DIR 解析到仓库根
- stairs env 冒烟: obs_groups_spec = {obs: 297, critic: 511}, reset/step 正常
- 8 个训练全部进入训练循环:
  flat ETA~6h / rough~11h / jump_flat~5h / jump_srl~5h /
  jump_vmc~7.5h / jump_srl_vmc~7.5h / stairs~13h / toe_walk~5h (GPU0, num_envs=1024)

## 后续计划

- [x] 8 个训练全部启动并进入循环
- [x] 修复 ROOT_DIR 路径 bug
- [x] 修复 stairs obs 维度 bug
- [ ] 监控训练 (TensorBoard: `bash shell/xqrobotV2/tools/tensorboard.sh`)
- [ ] 提交代码修复 (不推送)
