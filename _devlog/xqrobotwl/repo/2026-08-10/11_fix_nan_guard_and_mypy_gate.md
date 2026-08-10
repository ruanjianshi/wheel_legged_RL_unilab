# [11] 修复 NanGuard 测试失败 + make type (mypy/pyright) 门禁

**日期**: 2026-08-10
**来源**: 整理项目 ([[10_reorganize_per_claude_spec]]) 验证时发现的既有债: 5 个 NanGuard 测试失败 + `make type` 150 mypy 错 / 2 pyright 错
**关联**: [[10_reorganize_per_claude_spec]]

---

## 问题描述

`make test-all` 门禁不绿, 两类既有问题:

1. **5 个 NanGuard 测试失败** (tests/test_nan_guard.py + tests/base/test_np_env.py): caplog 捕获不到 nan_guard 的 WARNING, 断言 `len(warnings)==2` 得到 0。
2. **`make type` 150 mypy 错 + 2 pyright 错**: 跨 19 个文件, 以 `union-attr` (100, 全在 `RewardConfig | None` 的字段访问) 为主。

## 根因分析

### NanGuard (环境污染, 非代码 bug)
- 本机 shell `PYTHONPATH` 含 `/opt/ros/humble/lib/python3.10/site-packages` (ROS2)
- pytest 启动早期某模块导入 ROS `launch` 包 → `launch/logging/__init__.py` 模块级执行 `logging.setLoggerClass(LaunchLogger)`
- `LaunchLogger.__init__` 设 `propagate=False` → **之后创建的所有 logger** 都默认不传播 → caplog (挂 root handler) 永远捕不到
- 实测: `PYTHONPATH=.` 覆盖 ROS 路径后测试即过; `logging.getLoggerClass()` 在 pytest 里是 `launch.logging.LaunchLogger`
- **修复**: nan_guard 无自带 handler, 必须靠传播才能被捕获/显示 (否则真实训练 NaN 警告也静默消失) → 显式 `logger.propagate = True`

### mypy (配置类动态类型 + 惰性属性)
| 类别 | 数量 | 根因 | 修法 |
|---|---|---|---|
| union-attr | 100 | `_jump_cfg`/`_toe_cfg` 被基类首赋推断为 `XqRobotWLJumpRewardConfig \| None`, 子类访问自家字段 (h_cmd2/force_max/...) 失败 | 子类收窄注解 `_jump_cfg: <子类RewardConfig>` + `type: ignore[assignment]` + `if None: raise` 收窄 |
| assignment | 16 | dataclass 字段类型不变性: 子类 cfg 重定义 `commands/reward_config` | `# type: ignore[assignment]` |
| index | 11 | `RewardContext.gravity/dof_vel: ndarray \| None` 直接 `[:, ]` | `assert ctx.xxx is not None` (对齐 rewards.py:175 既有模式) |
| arg-type | 11 | `_jump_cfg` 传非 None 奖励函数 | 同 union-attr (收窄注解) |
| attr-defined | 10 | `env: object` 参数访问 `_cfg`; `SimBackend` 无 `_model` | `env: Any`; `# type: ignore[attr-defined]` (已 hasattr 守卫) |
| no-redef | 1 | toe_walk 两处分支重复注解 `_reward_fns` | 去掉第二处注解 |
| has-type | 1 | np3o 惰性 `_cost_buf` | 类级注解 |
| pyright×2 | 2 | cpo/np3o `compute_returns(last_obs)` 与基类 PPO `compute_returns(obs)` 参数名不匹配 | 改名 `obs` |

## 修改了什么

- `src/unilab/utils/nan_guard.py`: `logger.propagate = True` (恢复标准传播语义, 防御 ROS launch 劫持)
- **10 个 env 文件** 收窄 `_jump_cfg`/`_toe_cfg` 注解 + `if cfg.reward_config is None: raise ValueError(...)` (与基类错误契约一致) + `_provider`/`_state`/`ctx` 断言 + 配置字段 ignore:
  `xqrobotwl/{fall_recovery,backflip,single_leg,single_leg_move,toe_walk,jump,jump_srl,stairs,joystick}.py`
  `xqrobotV2/{toe_walk,jump,joystick,stairs}.py`
- `src/unilab/algos/torch/np3o.py`: 类级 `_cost_buf/_cost_accum/_cost_val_buf: torch.Tensor` 注解 + `compute_returns(last_obs→obs)`
- `src/unilab/algos/torch/cpo.py`: `compute_returns(last_obs→obs)`

## 验证方法

- `make format` (ruff) ✅
- `make type` ✅: mypy "Success: no issues found in 201 source files" + pyright "0 errors" (4 warning 为 motrixsim/mlx 平台缺失导入, 非错误)
- 全套 `uv run pytest`: **796 passed / 0 failed** (修复前 791/5)
- 环境创建冒烟: 缺 reward_config 时正确抛 `ValueError` (与基类一致), 非 AssertionError
- 修复前 vs 后: 150+2 → 0

## 遗留说明

- `PYTHONPATH` 里 ROS humble 路径是**环境级**污染, 非仓库问题; 若他处再出现 caplog 捕不到, 先查 `logging.getLoggerClass()` 是否为 LaunchLogger
- 4 个 pyright warning (motrixsim/mlx import) 是平台性缺失, 不阻断门禁

## 后续计划

- [ ] 提交 (用户自行 commit, 不推送)
