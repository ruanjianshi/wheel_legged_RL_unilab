# 参考文档: 跳跃姿态奖励 + 膝关节极限约束 (VMC) (2026-08-16)

> 问题 P4 (SRL+VMC): 髋外展 (hip_roll 0.611) 起跳 + 无下蹲蓄力; 以及三算法膝过伸超 ±0.85。
> 核心诉求: 跳跃过程应是"下蹲蓄力→上跳→收腿→落地", 髋不外展, 膝不越机械极限 (CLAUDE.md §7.5/§1.3)。
> 关键结论: ①外展轴 (hip_roll) 需要**侧向/对称姿态奖励**拉回默认角; ②反向下蹲 (countermovement) 是 RL 自然收敛的跳跃形式, 给够时序奖励即可; ③**虚拟腿长 L0 的可行域必须受膝关节机械极限约束**。

## 参考来源

1. **"Towards Low-Gravity Planetary Exploration using RL for Walking, Jumping, and In-flight Attitude Control"** (arXiv:2605.24643)
   - 跳跃姿态奖励: **landing position rewards** 把侧向/横向关节 (含四腿外展/内收轴) 拉回默认角; **lateral symmetry** 奖励同侧前后腿外展轴相似, **transversal symmetry** 奖励内外横向关节匹配 → 干净跳跃 + 预落地腿形。
2. **"Reinforcement learning of single legged locomotion"** (IROS 2013)
   - 学到的跳跃**自然收敛为 countermovement jump (先下蹲再起跳)**; 前跳髋膝共同出力, 后跳主要靠膝。→ 下蹲蓄力不需要硬编码, 给时序奖励就能学出。
3. **"Squat and tuck jump maneuver for single-legged robot with active toe joint using model-free DRL"** (2024)
   - 无路径规划的蹲跳/收腿跳; 系统研究奖励复杂度对跳跃行为的影响。
4. **"SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping"** (arXiv:2606.18625)
   - 与我们的 Wheeled-SRL 同名同范式: SLIP 前馈 + PPO 反馈 + 分阶段奖励; 双足实机跳高 15cm/跳远 20cm。**(建议下载全文细读, 对照我们的六态 FSM)**
5. **"Robust and Versatile Bipedal Jumping Control through Multi-Task RL"** (arXiv:2302.09450)
   - 分阶段奖励: pelvis roll/pitch 最小化 + **smoothing (电机速度/关节加速度/力矩消耗)** + action-change; 特别提到"防策略退化成站着不跳"的 episode 设计。
6. **Oehlke et al., "Template-based hopping control of a bio-inspired segmented robotic leg"** (IEEE BioRob 2016)
   - VMC 虚拟弹簧经**膝关节力矩**实现; **膝关节极限是核心约束** — MARCO Hopper II 膝是单侧伸展驱动, 重力作屈, 控制器必须在飞行期把虚拟刚度归零 (reflex) 并靠重力复位。
7. **CaT "Constraints as Terminations for Legged Locomotion RL"** (IROS 2024) + **second-order QP safety filter** (Cariou et al. 2026) + **"Robust Quadruped Jumping via DRL"** (arXiv:2011.07089)
   - 关节限位的三种做法: (a) 训练期 CaT (约束作为随机终止概率, Solo-12 关节限位违规 0.0%); (b) 运行时 QP 安全滤波/动作投影 (无需重训, 强制关节限位); (c) 朴素饱和不够, 跳跃需建模 torque-speed/power 极限。
8. **Ascento: A Two-Wheeled Jumping Robot** (ICRA 2019) + **"Jump Planning and Airborne Attitude Control of Bipedal Wheel-Legged Robots"** (IEEE 11076174)
   - 轮腿跳跃用 **W-SLIP 模型**规划; LQR/MPC/自适应 LQR 平衡; **空中姿态控制**; 扭转弹簧抵消自重提高跳跃高度。

## 核心方法

1. **外展轴姿态奖励 (P4 直接对症)**: 跳跃全程 (含 thrust/flight/landing) 对外展/内收轴加 (a) 默认角回归项 `−|roll − roll_default|` (b) 左右对称项 `−(roll_L + roll_R)²` (c) 侧向对称 (同侧前后腿)。
2. **反向下蹲 (CMJ) 奖励**: 触发后先奖"下蹲到位" (base_z 降到 crouch_target), 再奖"蹬伸" (vz 上升); 时序奖励 (crouch_depth + vertical_thrust 分窗) 即能学出 CMJ, 无需硬编码轨迹。
3. **膝关节极限 → L0 可行域裁剪 (膝过伸直接对症)**:
   - 由 VMC 逆运动学求出"膝角 ∈ [−0.85, +0.85]"对应的虚拟腿长区间 `[L0_min_knee, L0_max_knee]`;
   - 对 FSM 腿长参考和策略动作都裁剪到该区间 (`thrust_length` 若超出则截到 `L0_max_knee`);
   - 或按 CaT: 膝角超限作为随机终止概率; 或运行时动作投影 (QP/clip) 到可行域。
4. **跳跃期 smoothing + 姿态**: pelvis roll/pitch 最小化持续到腾空/落地; landing 后加 damping 项吸收冲击。
5. **W-SLIP/空中姿态**: 轮腿跳跃的 W-SLIP 规划 + 腾空期姿态控制 (空中收腿/预落地腿形), 改善"跳过 flight 态"问题。

## 可借鉴点 (对应我们 P4/膝过伸)

| 文献技术 | 我们的对应改法 |
|---|---|
| 侧向/对称姿态奖励 | `jump_srl_vmc.py` 跳跃期 (thrust/flight/landing) 加 `−|roll−0.1|−|roll+0.1|` + `−(roll_L+roll_R)²`, 或给 `jump_height`/`height_progress` 加 `roll_ok` 门控 (像 crouch_prep) |
| CMJ 时序奖励 | 已有 `crouch_depth`+`vertical_thrust`; 需**分窗不重叠** (先蹲满再蹬), 当前 SRL+VMC 触发即起跳 → 蹲窗奖励没生效, 检查 crouch 门控是否被 policy 跳过 |
| 膝限 → L0 裁剪 | 在 `vmc.py`/`jump_vmc.py` 求 `L0_max_knee` (膝 ±0.85), 对 `thrust_length` 与动作 `L0` 裁剪; 3 算法都受益 |
| CaT 关节限位 | 训练期加膝角终止概率 (低概率, 如超过限位 5%) 或在 `_compute_terminated` 保留膝限终止 (SRL 已有, VMC 没有) |
| smoothing (跳跃期) | 加 `joint_vel` 惩罚覆盖 thrust/flight, 抑制蹬伸乱摆 |
| W-SLIP 空中姿态 | 腾空期加姿态奖励 `up_z` 保持 (SRL+VMC 已有 jump_upright, 但 flight 态都跳过 → 先修 flight 态进入问题) |

## 与我们的差异

- 文献多为四足/单腿, 无轮; 我们是轮腿倒立摆 + 轮地接触, 腾空判定与落地冲击特性不同 (已有几何接触检测)。
- MARCO 膝是单侧驱动, 我们是双侧力矩 → 膝过伸是"力矩过大"而非驱动方向问题, 更该用 L0 可行域裁剪而非归零。
- 我们 VMC 的膝过伸主因是 `thrust_length=0.50` 超膝可行 L0, 裁剪参考比加惩罚更直接。

## 落地建议 (优先级 P4 + 膝过伸)

1. **膝限裁剪 (三算法通用, 机械安全最高优先)**: 计算膝 ±0.85 下的 `L0_max_knee`, 在 `_jump_leg_reference` (VMC 两变体) 与 `apply_action` 裁剪 L0; 纯PPO/SRL 在 `apply_action`/`step` 裁剪膝动作。改一处 VMC 参考即可覆盖 PPO+VMC/SRL+VMC。
2. **跳跃期外展惩罚 (P4)**: `jump_srl_vmc.py` 加 `lateral_posture` 奖励 (thrust/flight/landing 有效): `−(|roll_L−0.1|+|roll_R+0.1|)` scale≈10-20, 或给跳高奖励加 roll 门控。
3. **修"无下蹲"**: 检查 SRL+VMC 是否在 crouch 相被 policy 直接顶起 (VMC 腿长参考被残差抵消 / crouch 相 roll 外展顶高); 加强 crouch 相 L0 参考主导 (fb 0.35→0.2) 或加"蹲到位才给 vertical_thrust"门控 (vz>0 且 base_z<crouch_threshold)。
4. **flight 态进入修复**: 腾空判定后 FSM 应进 flight (当前 SRL+VMC 两次都跳过 2 → 轮没真正离地或判定太严), 检查 `_update_fsm_state` flight 条件 (vz<0 & z<dh+0.20) 与几何接触冲突。

**关联**: [[2026-08-16_jump_standing_stabilization_rl]] (P3), [[wheel_legged_lab]] (同构工程, VMC+六态跳跃)
