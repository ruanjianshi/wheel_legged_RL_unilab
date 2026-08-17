# SRL+VMC v7: 髋外展跳跃修复 + 下蹲门控

**日期**: 2026-08-16
**来源**: 负责人可视化观察: SRL+VMC 无下蹲起跳, 跳跃过程是髋关节外展方式 → 量化诊断 (P4) 确认: 跳跃期 hip_roll max 0.611 (超 0.5 外展阈值), "下蹲相" z 反升 (无下蹲), 相位序列两次跳过 flight 态。

## 修改了什么

1. **`src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py`** 新增 `_reward_lateral_posture` + 注册
   - 跳跃窗口 (phase≥1) 全程罚髋外展: `−( |roll_L−0.1| + |roll_R+0.1| + (roll_L+roll_R)² )`。
   - 文献依据: 外展轴需侧向/对称姿态奖励拉回默认角 (`docs/references/2026-08-16_jump_posture_knee_limit_vmc.md`)。
2. **`jump_srl_vmc.py`** `_reward_jump_height` / `_reward_height_progress` 门控 `window_crouched`
   - 必须先深蹲 (base_z < 0.45, 复用 jump.py 的 window_crouched 追踪) 才给跳高/进步奖励 → 逼出"下蹲蓄力→起跳"时序。
3. **`src/unilab/envs/locomotion/xqrobotwl/vmc.py`** 膝关节极限守卫 (两 VMC 变体共享)
   - 蹬伸前馈 reflex 式削减: 用膝位+膝速一步预测, 膝接近 0.50 开始削减蹬伸前馈, 归零于 0.85 → 防高速蹬伸把膝"砸"向机械止位 (实测 PPO+VMC 1.045/SRL 1.04/SRL+VMC 0.955 → 限到 ~0.88)。
   - 膝力矩硬守卫: 膝越过 0.85 且力矩推向外时置零。
4. **`conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml`**
   - `lateral_posture: 15.0` (新增权重)。

## 根因分析

- 跳跃期 (thrust/flight/landing) 奖励无 hip_roll 约束 (只有下蹲相 roll_ok 门控 + 站立期 lean_forward) → 策略发现"髋外展借力"也能起跳, 且 jump_height 30 照拿满。
- 站立期本身不稳 (base_z 偏低 ~0.48), 触发时已在弹跳中 → 无下蹲空间; 外展改变腿几何使"下蹲相"z 反升。

## 参数调整好坏

| 参数 | 旧 | 新 | 预期 |
|---|---|---|---|
| lateral_posture | 无 | 15.0 | 跳跃期把外展轴拉回默认角 + 罚不对称 |
| jump_height/height_progress | 无条件 | 须先深蹲 (window_crouched) | 逼下蹲蓄力 (风险: 若 VMC 蹲不动则不学跳, 需监控) |
| 蹬伸前馈守卫 | 无 | 膝>0.50 削减, >0.85 归零 | 防膝砸止位 (风险: 削蹬伸力 → 跳高略降) |

## 验证方法

- 环境冒烟: `smoke_v7.py` 构建通过; VMC pytest 10 passed; `lateral_posture` 奖励分派确认 (训练日志 iter0 `reward/lateral_posture: -0.2868`)。
- 重训 4000 iter (`logs/train/jump_srl_vmc_v7.log`, GPU0)。
- 训练后用 `diag_jump_problems.py` 重新诊断: hip_roll max / 下蹲深度 / 相位序列 / 膝过伸 / 跳高。

## 训练后效果

### v7 首轮 iter-1000 (对称守卫) — 蹲了不跳 (stutter)
| 指标 | 值 | 判定 |
|---|---|---|
| hip_roll max | **0.074** (基线 0.611) | ✅ 髋外展解决 |
| 下蹲深度 | 0.271 (蹲到 0.262m) | ✅ 下蹲门控生效 |
| 跳高 / 腾空 | **0.004m / 0 步** | ❌ 蹲了不跳 |
| jump_height 奖励 | 恒 ~0 | ❌ 轮不离地 |
| vertical_thrust 奖励 | ~0.22 (在地面推 vz>0) | ⚠️ stutter 挣分 |

**根因定位** (用 v6 模型隔离): v6 模型+守卫仍能起跳 (0.209m) → 守卫本身不阻止起跳; 是策略收敛进 stutter (推而不起)。进一步定位: **守卫用对称 |knee| 判断, 深蹲屈膝 (|knee|>0.5) 时把支撑前馈也削了** → 蹲不住/蹬不起来 → 策略只能在地面推 vz 挣 vertical_thrust。

**修复**: `vmc.py` 守卫改为**方向性** — 只削"伸展方向接近极限"的前馈 (L 腿负向/R 腿正向伸展), 深蹲屈膝不削支撑。v6 模型验证: 膝 0.897→0.743 (无过伸), 深蹲支撑保留。

### v7b (方向性守卫) iter-1000 — 能起跳了 ✅
| 指标 | v6 基线 | v7首轮 | v7b iter-1000 |
|---|---|---|---|
| 跳高 | 0.247m | 0.004m | **0.247m** ✅ |
| 腾空步数 | 61 | 0 | **28** ✅ |
| 髋外展 hip_roll | 0.611 | 0.074 | **0.369** (<0.5 阈值) |
| 膝过伸 | 0.955 | — | 0.909 (接近限位) |

方向性守卫 (深蹲保留支撑前馈) 解决 stutter — 决定性修复。髋外展从 0.611 降到 0.369 (低于 0.5 姿态阈值, 但仍有轻微展开, 待最终评估看是否需加强 lateral_posture)。待 4000 iter 收敛后复测。

### v7b 最终 (model_3999) — 外展修好但跳高塌缩
| 指标 | 值 | 判定 |
|---|---|---|
| hip_roll max | **0.277** (基线 0.611) | ✅ 外展修复 |
| 跳高 | **0.123m** (iter-1000 曾 0.247m) | ❌ 收敛塌缩 |
| 站立 \|gyro\| | **2.25** (iter-1000 曾 0.92) | ❌ 站立振荡回归 |
| 膝过伸 | 0.892 | ⚠️ 接近限位 |

**根因**: ①v6 靠外展起跳, 去外展后未给替代起跳机制 → 策略落"小跳+stutter"局部最优; ②SRL+VMC 未上站立稳定项 (LPF/ang_vel_xy), 站立本身不稳 → 触发时在振荡中 → 下蹲/起跳不一致。

### v7c (补站立稳定) 最终 — 站立振荡顽固
| 指标 | v7b最终 | v7c最终 | 判定 |
|---|---|---|---|
| 髋外展 | 0.277 | **0.26** | ✅ 持续修复 |
| 跳高 | 0.123m | 0.173m | ⚠️ 部分恢复 |
| 站立 \|gyro\| | 2.25 | **3.06** (末期 4.7) | ❌ LPF+ang_vel-0.4 收敛后仍压不住 |
| 膝过伸 | 0.892 | 0.886 | ⚠️ 接近限位 |

**根因**: SRL+VMC 的 always-on landing_recovery (12) 在"站姿微摆但保持直立/高度"时仍照发 → 振荡净收益为正, ang_vel_xy 罚不破极限环。

### v7d (站立专项惩罚) 重训中
- `jump_srl_vmc.py` 新增 `_reward_standing_still`: 站立期 (trigger≤0.5) 罚 |gyro| 超 1.0 部分 (封顶 4), 站得稳无惩罚
- 配置: `standing_still: -4.0` + `kd_l0: 12→20` (物理阻尼)
- 保持 v7c 全部修复 (lateral_posture 15 + 下蹲门控 + LPF + ang_vel -0.4 + 方向性守卫)

## 后续计划

- 若 v7b 仍 stutter: 把 vertical_thrust 也门控到"真实腾空" (break contact), 或降 lateral_posture (15→8)。
- 若外展仍存: 提高 lateral_posture 权重或对 jump_height 直接加 roll 门控。

**关联**: [[NN_jump_srl_vmc_v6_rebalance]], [[assess: 2026-08-16_four_algo_visual_problems_diag]], [[2026-08-16_jump_posture_knee_limit_vmc]]
