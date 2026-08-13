# [25] 修复: fall_recovery 键盘交互回放报 MissingConfigException (play 支持 cpo)

**日期**: 2026-08-12
**来源**: 用户运行 `bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh --keyboard`
报 `hydra.errors.MissingConfigException: Could not find 'task/xqrobotwl_fall_recovery_flat/mujoco'`
**关联**: [[24_overnight_v8_v82_delivery]] (交付 model_7000 后用户想交互回放验证)

---

## 根因分析

fall_recovery 是 **CPO** 任务 (配置在 `conf/cpo/`, 日志根 `rsl_rl_cpo`), 但:

1. `shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh` 硬编码 `--algo ppo`
   → play_interactive 到 `conf/ppo/` 找任务配置 → 找不到 (报 MissingConfigException)
2. `scripts/play/play_interactive.py` 的 algo 注册表**根本没有 cpo**:
   - `SUPPORTED_INTERACTIVE_ALGOS` 无 cpo → `--algo cpo` 会被 argparse 拒绝
   - `_CONFIG_ROOT_BY_ALGO` 无 cpo → 即使传 cpo 也无配置根映射
   - 主流程 `if algo in ("ppo", "np3o")` 分支 (rsl_rl 回放会话) 不含 cpo → 会掉到
     `else: raise ValueError("Unsupported interactive playback algo: cpo")`

## 修改了什么

| 文件 | 改动 |
|---|---|
| `scripts/play/play_interactive.py` | ①`SUPPORTED_INTERACTIVE_ALGOS` 加 `"cpo"`;②`_CONFIG_ROOT_BY_ALGO` 加 `"cpo": "cpo"`;③主流程分支 `("ppo", "np3o")` → `("ppo", "cpo", "np3o")` (CPO 继承 PPO, 同用 OnPolicyRunner/RslRlVecEnvWrapper 回放会话) |
| `shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh` | `--algo ppo` → `--algo cpo` (fall_recovery 是 CPO) |

## 验证方法 (无 GUI 全链路测试)

| 环节 | 结果 |
|---|---|
| 配置组合 `_compose_interactive_config('cpo', ...)` | ✅ task=XqRobotWLFallRecoveryFlat, algo_log_name=rsl_rl_cpo, num_constraints=2 |
| checkpoint 解析 (rsl_rl_cpo 最新) | ✅ 解析到 `logs/rsl_rl_cpo/.../model_8999.pt` |
| actor 输入维度推断 | ✅ 297 |
| env 构建 (backend adapter cpo) | ✅ obs["obs"].shape=(1,297), num_action=8 |
| ruff format/check | ✅ |
| shell 语法 (bash -n) | ✅ |

## 效果

- `bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh --keyboard` 现在能正常
  组合 cpo 配置、解析 checkpoint、构建 env、打开 MuJoCo viewer 交互回放
- 附带修复: 任何 CPO 任务的 play/键盘回放都可用 (目前仅 fall_recovery 用 CPO)

## 注意 (对用户)

- 脚本默认加载 **最新 run** (v8.4 03-12-53, 恢复崩), 交互回放请显式指定**交付 run**:
  `bash shell/xqrobotwl/fall_recovery/eval_ppo_fall_recovery.sh 2026-08-12_01-58-14_mujoco --keyboard`
  (model_7000, 3/4 姿态恢复 + 站立 5.6-7.2s)
- 仰卧 (pose 0) 恢复率 0% (v8.4 已知局限, 见 #24)

## 后续计划

- [ ] 用户交互回放交付 model_7000, 观察 3 姿态恢复效果
- [ ] (可选) 脚本默认 run 提示: 交付 run 与最新 run 区分
