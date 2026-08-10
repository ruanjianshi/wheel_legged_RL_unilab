# 八任务评估体系 (Assess)

> 按 CLAUDE.md 开发规范构建的完整评估体系 —— **数据优先 (CSV 是评估依据) + 达标判定 (✅/❌) + 姿态反推**。
> 覆盖八任务 §7.0 通用评估 / §7.x 各任务达标标准 / §1.5 数据优先 / 附录 A 核心指标。

## 快速开始

```bash
# 列出八任务
uv run python _devlog/assess/runner.py --list-tasks

# 平地滚动行走评估 (默认 decoupling 套件 + 达标判定)
uv run python _devlog/assess/runner.py -t walk_flat -r <run> -c model_9999.pt

# 完整评估 + 姿态数据 + 报告
uv run python _devlog/assess/runner.py -t walk_flat -r <run> -c model_9999.pt \
    -s full --dump-pose --report

# 跌倒恢复 (固定倒地姿态 0=仰卧, 20 episodes)
uv run python _devlog/assess/runner.py -t fall_recovery -r <run> --pose 0 --num_envs 20

# 跳过达标判定 (只出指标)
uv run python _devlog/assess/runner.py -t jump -r <run> -c model_9999.pt --no-verify
```

> 注: 评估走 MuJoCo 离屏后端, 普通 `uv run python` 即可; 需要渲染视频时用
> `uv run mjpython` + `tools/xqrobotwl/render_*.py`。

## CLI 参数

| 参数 | 说明 |
|---|---|
| `-t, --task` | 任务 key: walk_flat / toe_walk / walk_rough / jump / backflip / single_leg / fall_recovery / stairs |
| `-r, --run` | run 目录名 (在任务 log_root 下, 如 `2026-08-10_xx_mujoco`) |
| `-c, --ckpt` | checkpoint 文件名 (默认最新 model_*.pt) |
| `-s, --suite` | 行走类场景套件: decoupling / full / standing |
| `--num_envs` | 并行 env / episode 数 (跌倒恢复建议 20) |
| `--max_steps` | 每 episode 最大步数 |
| `--pose` | 跌倒恢复固定倒地姿态 0-3 (-1 随机) |
| `--jump_every` | 跳跃/后空翻触发周期 (步) |
| `--dump-pose` | 导出姿态数据 CSV → `logs/pose_data/` |
| `--report` | 生成 Markdown 评估报告 + JSON |
| `--no-verify` | 跳过 §7.x 达标判定 |

## 八任务注册 (tasks.py)

| 任务 | env 注册名 | conf | algo | §7.x 达标 |
|---|---|---|---|---|
| walk_flat | XqRobotWLWalkFlat | ppo/xqrobotwl_walk_flat | ppo | 追踪<0.1, 存活≥95%, 侧移 Vy>0.25, 微动平衡 |
| toe_walk | XqRobotWLToeWalkFlat | ppo/xqrobotwl_toe_walk_flat | ppo | 高度≈0.52±0.05, 抬腿平缓, 追踪 |
| walk_rough | XqRobotWLWalkRough | ppo/xqrobotwl_walk_rough | ppo | 存活≥90%, 高度波动小 |
| jump | XqRobotWLJumpFlat | ppo/xqrobotwl_jump_flat | ppo | 成功率≥90%, 跳出高度, 有腾空 |
| backflip | XqRobotWLBackflipFlat | ppo/xqrobotwl_backflip_flat | ppo | 翻转完成率≥90%, 落地站立 |
| single_leg | XqRobotWLSingleLegFlat | ppo/xqrobotwl_single_leg_flat | ppo | 单腿保持≥5s, 倾斜行走追踪 |
| fall_recovery | XqRobotWLFallRecoveryFlat | cpo/xqrobotwl_fall_recovery_flat | cpo | 恢复率≥80%, 最长站立≥0.5s, 漂移<0.5m |
| stairs | XqRobotWLStairs | np3o/xqrobotwl_stairs | np3o | 上台阶成功率≥90% |

## 组件

```
assess/
├── tasks.py       # 八任务注册表 + §7.x 达标阈值
├── engine.py      # 通用确定性 rollout 引擎 (建env/加载policy/跑episodes/逐行采集)
├── metrics.py     # 附录 A 核心指标 + 追踪/稳定/运动质量 (StepSample/Trace 载体)
├── scenarios.py   # 场景模型 + 行走套件 (decoupling/full/standing)
├── verify.py      # §7.x 达标判定 → 逐项 ✅/❌ + 总体
├── report.py      # 输出: stdout 摘要 + results/<task>/<session>/metrics.json + reports/<task>/eval.md
├── pose.py        # 姿态数据 CSV 导出 (§1.5.1, 26 列, 两位小数) → logs/pose_data/
├── infer.py       # 姿态反推统计 (§1.3/§1.5.2, 复用 infer_pose_from_csv 判定)
├── runner.py      # 统一 CLI
└── eval/          # 八任务评估模块 (各实现 §7.x)
```

## 数据优先闭环 (§1.5)

```
runner 评估 → engine.collect_step 逐行采集 (26 列)
  ├─ metrics.py   → 附录 A 指标 + §7.x 判定
  ├─ pose.py      → logs/pose_data/<task>_<ckpt>.csv (两位小数)
  └─ infer.py     → 姿态分布 (站立/下蹲/前倾/转圈/… 帧数+时长+占比)
```

报告时 **数据优先**: 附 CSV 路径 + 关键指标, 视频作补充 (CLAUDE.md §1.5.3)。

## 与既有工具的关系

- 复用 `tools/xqrobotwl/verify_jump.load_actor` (策略加载) / `infer_pose_from_csv` (姿态判定)
- `tools/xqrobotwl/eval_fall_recovery.py` 仍是独立的跌倒恢复评估入口 (runner 用同一套 rollout 模式)
- 训练监控 `shell/xqrobotwl/tools/monitor_training.sh` 每 1000 iter 提醒跑本 runner (§1.2)

## 达标判定依据

- 阈值来源: CLAUDE.md §7.2-7.9 各任务达标标准 + 附录 A 核心指标 + §1.4 微动平衡
- 长时评估 (§7.0): 站立≥10s / 行走≥30s / 动作≥10 次 / 恢复每姿态≥20ep 在 eval 模块参数化
