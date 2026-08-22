# 站立按键切换独轮车式：FSM 与过渡训练

**日期**：2026-08-17

## 目标

机器人正常双轮站立启动；按 `H` 后平滑进入横躺独轮车式并保持，再按 `H` 返回双轮站立。旧 `XqRobotWLSingleLegUnicycle` 直接从横躺姿态 reset，不具备这个过渡能力。

## 根因

1. 交互层虽然把 `H` 键写入 `commands[:, 4]`，独轮环境却把该维当作固定高度，按键没有模式语义。
2. 环境 reset 直接使用横躺 `Rx(90°)` 姿态，策略从未观察正常站立或过渡状态。
3. 六个腿关节同时插值会塌低；真实动力学需要先展开左支撑腿、卸载并抬起右轮，再展开右侧配重腿。
4. 初版过渡按模式进度缩小左轮扭矩，使旧策略在最需要抑制俯仰时几乎没有轮控制权。
5. 过渡期角速度与横向运动是必要动作，旧 damping/lateral 惩罚全程生效会阻止侧翻。

## 实现

- 命令定义改为 `[vx, vy, vyaw, tsk, unicycle_trigger]`。
- 新增四态 FSM：`-1 站立 → 0 进入过渡 → 1 独轮保持 → 2 返回过渡 → -1`。
- `H` 使用锁存 ON/OFF；交互模式下关闭随机命令重采样，防止覆盖按键。
- 模式进度使用 smoothstep；站立和独轮端点保持旧 324/351 维网络结构。
- 独轮态观测末维保持旧 checkpoint 熟悉的 `0.6776`，可迁移既有 8 秒平衡策略。
- 过渡期开放六腿关节残差；一进入过渡即开放完整左轮俯仰控制。
- 腿轨迹改为左支撑腿先展开、右配重腿后展开；返回时顺序自动反转。
- 终止判据分阶段：站立检查直立，过渡允许必要滚转，独轮态严格检查方向、左轮支撑与右轮离地。
- `mode_complete` 只在真实姿态和双轮接触条件同时满足时发放，避免倒地刷奖励。
- 新增严格评估工具 `tools/xqrobotwl/eval_single_leg_unicycle_transition.py`。

## 验证与训练

- `ruff check`: passed
- `tests/envs/locomotion/xqrobotwl/test_xqrobotwl_single_leg_unicycle.py`: **10 passed**
- v3 同步插值训练到 `model_1000`：100% 进入 FSM，但真实独轮姿态 0%，平均存活 3.02/7.00 s；作为失败对照保留。
- v4 使用分阶段腿轨迹，从 v3 `model_1000` 继续微调到 `model_1999`；仍为 100% 进入 FSM、0% 真实独轮姿态，平均存活 3.02/7.00 s。
- v6 引入后半程逆向 reset 课程，训练日志首次稳定出现非零 `mode_complete`，证明局部过渡和真实终点可以学到；但原独轮 checkpoint 在普通站立端点约 0.53 s 即倒，端点技能发生冲突。

## v7/v8：保留成熟站立专家

为避免重新学习已有的双轮平衡能力，将 actor 每帧前 33 维和 critic 每帧前 36 维严格对齐 `XqRobotWLWalkFlat`。首次 warm-start 将 297 维连续复制到 324 维，错误地跨越了 9 帧边界；修复工具改为逐历史帧散射：`33→36`（actor）、`36→39`（critic）。

另外发现仅复制网络仍不够：把共享执行器从 WalkFlat 的腿 `kp=60` 改成独轮所需 `kp=300/kd=10`，会改变按 H 前的子步闭环动力学。最终方案是：

- 保持 XML 原始执行器不变，站立态动作和动力学与 WalkFlat 完全一致；
- 过渡/独轮态把 `kp=300/kd=10` 期望扭矩换算为原位置执行器目标；
- 腿扭矩限制为 ±30 N·m，消除中间姿态的 QACC 数值爆炸；
- 单轮速度反馈只按模式进度渐入，不污染双轮站立；
- reset 首帧显式携带对应 FSM/进度，避免中间物理姿态被错误标记为站立。

修复后的未训练 warm-start 严格站立结果：20/20 环境完整运行 7 s，存活率 100%。128 环境 × 20 iterations 预检无 QACC/NaN，并持续出现非零 `mode_complete`。正式 v8c 使用 60% 后半程逆向 reset、40% 普通站立 reset 和 50% 站立触发样本同步训练：

```text
warmstart: logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/
           warmstart_transition_v7_walk_framed/model_0.pt
v8c: logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/
     2026-08-17_20-23-40_mujoco_transition_v8c_walk_reverse_safe_2000
```

训练期间必须分别验收两个分布：`hold_steps=0` 验证按 H 前站立不遗忘；完整 `stand 7 s → H → unicycle hold` 验证端到端物理转换。仅凭 FSM 进入率或课程分布的 `mode_complete` 不宣称完成。

v8c `model_1000` 严格结果：

| 验收项 | 结果 |
|---|---:|
| 按 H 前双轮站立 7 s | 20/20 |
| 进入独轮 FSM | 20/20 |
| 达到真实独轮姿态 | 0/20 |
| 完整 9 s 端到端存活 | 0/20 |
| 平均存活 | 5.02/9.00 s |
| 左支撑轮贴地率 | 90% |
| 右自由轮离地率 | 5% |

结论：成熟站立技能已完整保留，但 `transition_reset_min_progress=0.6` 只学到了过渡后半程，尚未把右轮卸载动作桥接到站立端点。因此从 model_1000 开始第二阶段 v8d，将逆向 reset 最小进度放宽到 0.3：

```text
logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/
2026-08-17_20-32-34_mujoco_transition_v8d_walk_reverse_mid_1000_to_2500
```

v8d `model_2499` 仍保持 20/20 双轮站立，但端到端真实独轮为 0/20。新增过渡轨迹诊断后确认：原轨迹的右轮在整个过渡中没有稳定卸载，FSM 到时后立即因自由轮落地终止。

## v9：三段式物理过渡

将轨迹重构为：

1. 右腿收膝到 `[-0.10, -0.61, -0.87]`，先把自由轮抬离地面；
2. 保持右腿收起，左腿展开并让机身滚到左轮单支撑；
3. 横滚稳定后再展开右腿到配重姿态。

诊断验证右轮过渡中最高达到约 0.30 m，横滚最佳对齐达到 0.98，证明“抬轮”和“侧翻”两个物理子动作均已实现。随后暴露两个工程限制：任务 episode 原上限 8 s 会截断延长过渡；统一 30 N·m 腿扭矩不足以让右髋 roll 跟踪 -1.50 rad。修复为：

- `max_episode_seconds: 8 → 20`；
- `transition_time: 2 → 7 s`；
- 右髋 roll 限幅 60 N·m，其余腿保持 30 N·m；90 N·m 实测会破坏横滚，未采用；
- 评估器新增过渡最大轮高、最佳横滚、独轮态实际腿角与右轮高度。

旧 v8d 策略在新轨迹上可达到横滚对齐 0.97，但右髋实际仅约 -1.08 rad，仍无法稳定完成配重展开，因此必须针对新轨迹重训。v9 从 v8d model_2499 继续：

```text
logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/
2026-08-17_20-49-12_mujoco_transition_v9_tuck_roll_deploy_2499_to_3499
```

训练目录：

```text
v3: logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/
    2026-08-17_19-38-53_mujoco_transition_v3_warm_5000
v4: logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/
    2026-08-17_19-48-28_mujoco_transition_v4_staged_resume_1000
```
