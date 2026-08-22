# SRL+VMC v9/v10: 奖励、触发与关节安全重训

**日期**: 2026-08-17  
**来源**: v8e5 最终模型虽然达到约 0.31 m 跳高，但重复跳跃出现较大漂移、落地失败和膝关节越限；进一步代码审查还发现奖励与评估实现错误。

## v8e5 问题复现

最终检查点：

```text
logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/
2026-08-17_12-50-27_mujoco_v8e5_10000/model_9999.pt
```

20 次重复跳跃：

| 指标 | v8e5 结果 |
|---|---:|
| 真正离地 | 19/20 |
| 恢复站立 | 18/20 |
| episode terminated | 2/20 |
| 总成功率 | 90% |
| 跳高 | 0.346 ± 0.069 m |
| 恢复窗口漂移 | 0.715 ± 0.819 m |
| 空中轮速峰值 | 10.9 rad/s |

异常样本包括：一次仅跳 0.149 m、却漂移 4.057 m；另一次跳到 0.476 m 后未恢复并终止。

蹬伸诊断结果：

```text
站立 z=0.555 m
跳高=0.319 m
最大向上速度=3.53 m/s
最大 VMC 合力命令≈1030 N
膝角范围: L=[-0.98, 0.80], R=[-0.80, 0.95] rad
XML 机械极限: ±0.873 rad
```

结论：v8e5 的高度优势伴随过强冲量、膝止位碰撞和落地可靠性下降，下一版不再继续追求最大高度，而是优化受约束的安全跳跃。

## 根因

### 1. `anti_drift` 双重负号

`_reward_anti_drift` 原本返回负值，配置又使用 `anti_drift: -3.0`，公共 reward dispatch 再执行 `reward × scale`，最终把漂移变成了正奖励。

```text
(-drift_penalty) × (-3.0) = positive reward
```

训练日志中 `reward/anti_drift` 为正，和大漂移行为一致。

### 2. SRL+VMC 的 `height_progress` 静默失效

高度进展奖励需要更新前的 `episode_prev_max_height`。纯 SRL 会保存该状态，但 SRL+VMC 继承的 `jump_vmc.update_state` 直接更新 `episode_max_height`，导致进展量恒为零。

### 3. 左右轮接触被错误复制

训练环境和 `eval_jump_full.py` 都只读取左轮世界坐标，然后把同一个接触值复制到左右轮。单轮卸载可能被误判为整机腾空或落地，污染：

- `jump_height` / air gating；
- `had_air` 与 `landing_timer`；
- 落地恢复奖励；
- 起跳、落地和恢复评估。

评估脚本还没有把 `terminated` 写入结果行，因此打印值可能恒为 `False`；起跳/落地也没有连续帧防抖。

### 4. 蹬伸强度超过机械约束

v8e5 使用 `thrust_kp_scale=3.0`、`thrust_ff_scale=3.5` 和很低的蹬伸阻尼。位置/前馈守卫只能撤掉继续外推的力，无法耗散已经积累的膝伸展动量，因此仍会越过软关节限位。

## v9 修改

### 奖励与状态

1. `_reward_anti_drift` 改为返回非负惩罚幅值，由负 scale 统一决定符号。
2. `jump_vmc.update_state` 在更新最高点前保存 `episode_prev_max_height`。
3. 新增 `knee_limit` 连续惩罚：膝角超过 0.72 rad 后逐渐增强。
4. SRL+VMC 在 `|knee| > 0.90 rad` 时终止，防止策略利用 MuJoCo 软止位。
5. 奖励重新平衡：
   - `anti_drift`: -3 → -5
   - `height_progress`: 20 → 10
   - `jump_event`: 100 → 60
   - `landing_soft`: 8 → 12
   - `landing_recovery`: 8 → 12
   - `knee_limit`: -15

### 双轮接触与评估

1. `xqrobotwl.xml` 和 `xqrobotwl_vmc.xml` 新增 `right_wheel_world_pos`。
2. 环境分别计算左右轮几何接触。
3. `eval_jump_full.py` 保存真实 `terminated`。
4. 起跳必须连续 3 步双轮离地；落地必须连续 3 步双轮接地，过滤载荷转移造成的短脉冲。

### 安全 VMC 参数

参数扫描选定：

```yaml
kd_l0: 30.0
thrust_length: 0.49
thrust_kp_scale: 2.5
thrust_ff_scale: 2.0
knee_guard_start: 0.35
knee_guard_limit: 0.82
knee_brake_start: 0.0
knee_brake_kd: 4.0
```

旧策略在该控制器下的验收结果约为：

| 指标 | v8e5 控制器 | v9 安全控制器 |
|---|---:|---:|
| 跳高 | 0.328 m | 0.24–0.26 m |
| 最大膝角 | 0.952 rad | 约 0.77 rad |
| 是否超过 ±0.873 rad | 是 | 否 |

安全控制器牺牲约 7–9 cm 极限高度，但消除了测试中的膝关节越限。策略重训后是否能在安全约束内恢复部分高度，需要以最终检查点评估为准。

## 验证

- `ruff check`: passed
- `tests/envs/locomotion/xqrobotwl/test_xqrobotwl_jump_vmc.py`: **14 passed**
- 30 iterations smoke training: completed
- `anti_drift` 在训练日志中已为负奖励
- `knee_limit` 在接近限位时生效
- 逐步检查确认 `height_progress` 在真实上升段连续 18 步非零；训练表格显示 `0.0000` 是跨 1024 环境平均及四位小数格式造成，不是再次失效

## 正式训练

配置：MuJoCo、PPO、1024 environments、10000 iterations、seed=1。

```text
run_suffix=fix_v9_safe_10000
logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/
2026-08-17_18-00-06_mujoco_fix_v9_safe_10000
```

v9 训练到 `model_5000.pt` 后停止。确定性回归显示跳高约 0.124 m，但 5/5 尝试均因深蹲或落地阶段膝关节越限终止，因此不能作为可用模型。

## v10 二次根因与修复

v9 的轨迹复查发现四个剩余问题：

1. 持续按住触发键时，FSM 回到 idle 后会再次启动跳跃，而不是一次按键对应一次动作。
2. `feedback_gain=0.35` 的策略残差仍可抵消 FSM 腿长参考，导致下蹲过深、落地前继续收腿。
3. 原膝关节制动只覆盖伸展方向，无法抑制深蹲方向的止位撞击。
4. `jump_event_threshold=0.72 m` 高于当前安全控制器可稳定达到的高度，训练中事件奖励长期为零。

v10 实现：

- 将原始触发转换为 rising-edge 锁存请求；按住按键只执行一次，松开后才重新武装。
- 下蹲阶段强制 `L0 >= 0.28 m`，预落地/落地阶段强制 `L0 >= 0.40 m`，策略残差不能覆盖安全参考。
- 在 `|knee| > 0.68 rad` 且关节继续向任一机械止位运动时施加耗散制动。
- `feedback_gain: 0.35 -> 0.15`，`init_noise_std: 0.3 -> 0.2`，`entropy_coef: 0.005 -> 0.001`。
- 将跳跃事件阈值调至可达的 `0.64 m`；v10 smoke 日志已出现非零 `reward/jump_event`。
- 修复重复跳跃评估：恢复成功必须持续到完整 recovery window 末尾，不能在短暂站稳后忽略延迟跌倒。

验证：

- `ruff check`: passed
- SRL+VMC 跳跃测试：**16 passed**
- v8e5 旧策略接入 v10 控制层：初次落地成功，5 次短窗口无膝越限；严格长窗口暴露延迟跌倒，证明新评估口径必要。

## v10 warm-start 训练

以 v8e5 的 actor/critic 为起点，但使用 v10 模板保留全新的优化器状态，并将动作标准差从旧 checkpoint 的约 `1.63` 重置为约 `0.20`：

```text
warmstart:
logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/warmstart_v10_safe/model_0.pt

training run:
logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/
2026-08-17_19-00-39_mujoco_fix_v10_edge_safe_warm_5000
```

中间确定性评估（20 次重复跳跃，严格完整恢复窗口）：

| checkpoint | 真跳 | 长窗口恢复 | 终止 | 跳高 | 漂移 |
|---|---:|---:|---:|---:|---:|
| model_1000 | 20/20 | 17/20 | 0/20 | 0.240 ± 0.062 m | 0.401 ± 0.281 m |
| model_2000 | 20/20 | 19/20 | 0/20 | 0.235 ± 0.057 m | 0.391 ± 0.290 m |
| model_3000 | 20/20 | 17/20 | 1/20 | 0.247 ± 0.076 m | 0.233 ± 0.203 m |
| model_4000 | 20/20 | 18/20 | 0/20 | 0.241 ± 0.060 m | 0.433 ± 0.395 m |
| model_4999 | 20/20 | 18/20 | 1/20 | 0.237 ± 0.059 m | 0.266 ± 0.375 m |
| model_6000 | 19/20 | 16/20 | 1/20 | 0.256 ± 0.077 m | 0.643 ± 0.557 m |
| model_6998 | 20/20 | 19/20 | 0/20 | 0.264 ± 0.066 m | 0.228 ± 0.170 m |

`model_6998.pt` 是最终推荐检查点：恢复可靠性与 model_2000 同为 95%、无终止，同时跳高从 0.235 m 提高到 0.264 m，漂移从 0.391 m 降至 0.228 m。model_6000 曾退化到 80% 总恢复率，说明训练并非单调改善，最终选择必须依赖严格重复评估。

```text
logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/
2026-08-17_20-24-43_mujoco_fix_v10_edge_safe_resume_4999_to_7000
```

## 完成后评估清单

1. 对最终 model_6998 跑更多随机种子与硬件扰动评估，确认 20 次样本外的稳定性。
2. 报告跳高、真正离地率、恢复率、终止率、恢复漂移和角速度异常值。
3. 运行 thrust profile，确认两侧膝角均不超过 ±0.873 rad。
4. 使用修复后的双轮接触评估起跳、落地和恢复事件。
5. 更新论文表格、轨迹图和最终性能图；不要复用 v8e5 的评估数值。

**关联**: [[NN_jump_srl_vmc_v8e5_clean_jump]], [[NN_jump_srl_vmc_v8e_rootfix]]
