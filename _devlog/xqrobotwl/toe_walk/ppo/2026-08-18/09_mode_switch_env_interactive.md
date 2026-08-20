# 09 — 双模式点足行走: 环境/交互/课程实现 (站立⇄抬腿, mode 命令通道)

## 日期
2026-08-18

## 来源
老板需求 (2026-08-18): 默认站立姿态 → 键盘按键切换成点足抬腿模式 → 指令追踪 (前进/后退/侧移/转向) → 按键切回站立。
方案已由老板审阅批准: [开发框架](../../../../docs/plans/2026-08-18_toe_walk_mode_switch_framework.md) + [参考文档](../../../../docs/references/2026-08-18_mode_switching_multi_skill.md)。

## 修改了什么

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/toe_walk_mode.py` | **新增** 双模式环境 `XqRobotWLToeWalkModeEnv` (继承 toe_walk flat env): commands 4D→**5D** [vx,vy,vyaw,tsk,**mode**], mode∈{0站立,1抬腿}; 观测 34→35/帧 (actor 306→315 dim, critic 342); 奖励按 env 的 mode 掩码门控 (抬腿项×mode, stand_still×(1-mode)); **lift_symmetry** 新奖励 (摆动相轮离地 L/R 滑动均值 EMA 差惩罚, 防单侧塌缩); 模式课程 Stage1 纯站立(0-2k)→Stage2 纯抬腿(2k-5k)→Stage3 随机切换(5k-10k), mode 每 6s 重采样 |
| `conf/ppo/task/xqrobotwl_toe_walk_mode/mujoco.yaml` | **新增** 双模式任务配置: action_scale 0.18, stand_still -5.0, lift_symmetry -20.0, mode_curriculum_iters [2000,5000], symmetry_decay 0.98 |
| `src/unilab/envs/locomotion/xqrobotwl/__init__.py` | +1 行: 导出 toe_walk_mode 环境 |
| `src/unilab/visualization/interactive_playback.py` | KeyboardCommander 加 `toe_mode` (H 键锁存) + `task_is_toe_mode` + toggle_toe_mode(); describe() 显示当前模式 |
| `scripts/play/play_interactive.py` | toe_walk_mode 任务: **H 键 = 站立/点足抬腿切换** (写 commands[:,4]); 主循环命令注入分支、legend、handler; 其余任务行为不变 |
| `shell/xqrobotwl/toe_walk_mode/train|eval_ppo_toe_walk_mode.sh` | 新增训练/评估启动脚本 (平台自适应 macOS/Linux) |
| `_devlog/assess/tasks.py` | 注册新任务 `toe_walk_mode` (obs_dim 315) |

## 训练后效果 (待训练完成评估)

- 冒烟 (2 iter, 8 env, GPU1): 全链路通过; Stage1 reward log 中抬腿四项=0 (mode 门控正确)
- 训练启动: 1024 envs × 10000 iter, log `logs/rsl_rl_ppo/XqRobotWLToeWalkMode/<ts>/`

## 参数调整好坏
- `lift_symmetry=-20` (摆动相 EMA L/R 差): 新引入, 针对 8-18 验证发现的单侧塌缩根因 (最新模型 L=0/R=12)
- 课程 [2000,5000] iter: 仿照 toe_walk v1 教训 (ramp 太陡崩), 站立/抬腿先分练后混合
- stand_still -5.0: 站立模式微动平衡 (零指令不动), 权重从 walk 系经验起步

## 根因分析
8-18 交替/对称验证证明: 纯抬腿配方无左右对称约束 → 训练后期塌缩到单侧抬腿转圈或纯轮滚。双模式需求进一步要求站立/抬腿共存于单策略, 需 mode 条件化 (Uni-Match/so100 思路) + 分阶段课程避免多目标冲突。

## 验证方法
1. ruff 全过; 2. env 构建 obs 315/critic 342/commands 5D; 3. mode=0 vs mode=1 reward log 门控; 4. 训练短冒烟 2 iter; 5. 正式训练中每 1k iter 评估 (待)。

## 后续计划
- 训练完成后: verify_toe_walk_mode.py 确定性评估 (站立→抬腿→追踪→站立 序列) + 姿态 CSV + 渲染视频
- 达标后备份 toe_walk_mode_v1

## 关联日志
- [08_verify_alternation_symmetry](2026-08-18/08_verify_alternation_symmetry.md) — 需求根因 (单侧塌缩)
- 框架/参考文档: `docs/plans/2026-08-18_...` / `docs/references/2026-08-18_...`