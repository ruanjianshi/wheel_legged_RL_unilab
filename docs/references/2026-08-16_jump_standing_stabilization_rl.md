# 参考文档: 站立振荡/无指令自跳 的强化学习解法 (2026-08-16)

> 问题 P3 (SRL): 跳完一次后 trigger=0 仍持续"蹲-起"振荡 (\|gyro\| mean 2.05 / max 5.2), 站立姿态偏差大。
> 核心诉求: 无指令时应"站立微动平衡" (CLAUDE.md §1.4), 而不是被策略自持极限环占据。
> 关键结论 (文献): **"站立"不会从移动/跳跃奖励里自然涌现, 必须显式奖励 + 抑制高频动作抖动 + 模式分离。**

## 参考来源

1. **Siekmann & Peng et al., "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"** (arXiv:2011.01387)
   - 周期性奖励组合 + **standing cost**: 命令落在站立区时, 额外加"动作差 cost + 脚对称 cost"; 权重系数 `ω` 随站立/行走模式调制 (`q_stand = 1 − exp(−(err_sym + 20·q_action_diff))`)。
2. **NSF bipedal locomotion work (par.nsf.gov/biblio/10600767)** — 行走后加"站立子阶段"课程: 随机行走一段时间后发站立命令; 站立时 (a) 参考运动切到标称站立姿态 (b) 提高 smoothing 权重 (电机速度 + 动作变化惩罚)。
3. **Gait-Conditioned RL (arXiv:2505.20619, "Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion")**
   - 观测里加 **one-hot gait ID** (standing/walking/running/transition) + **gait-conditioned reward routing** (按模式路由不同奖励, 屏蔽跨模式奖励干扰)。单一循环策略内可稳定切换多步态。
4. **"Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking"** (IEEE) — 站立稳定评估的重访, 强调站立奖励设计而非依赖涌现。

## 核心方法 (文献共识)

1. **站立必须显式奖励**: 加 standing region 检测 (命令≈0 时), 发 standing cost 惩罚 (动作差 / 不对称 / 速度)。
2. **动作低通滤波 (LPF)**: 策略输出后 Butterworth 4 Hz LPF, 消去高频抖动。消融: 去掉 LPF → "jittering motion"。**注意** 文献明说 LPF 只压高频, 整体振荡 (大关节速度) 仍需 smoothing 奖励。
3. **smoothing 奖励**: 电机速度惩罚 + 动作变化 (action-change) 惩罚。过强会"站着不探索" → 需与 LPF 配合调。
4. **模式分离**: gait ID / gait-conditioned reward routing — 让跳跃奖励与站立奖励互不干扰, 避免跳跃行为泄漏进站立 (我们 SRL 的核心病因)。
5. **站立命令课程**: 训练里明确穿插"站立期", 让策略把站立当独立模式学。

## 可借鉴点 (对应我们 P3)

| 文献技术 | 我们的对应改法 |
|---|---|
| 显式 standing cost (region-gated) | 在 jump.py/jump_srl.py 加 `standing` 检测 (trigger≤0.5): 罚 \|gyro\|、\|linvel_xy\|、关节速度; 加权重档位 (站立时强化) |
| 动作 LPF | 策略输出后加一阶/Butterworth 低通 (如 4-8Hz), 消 SRL 自持高频摆腿 |
| smoothing 奖励 | 强化 `joint_action_rate` / 加 `joint_vel` 惩罚 (仅站立期), 防蹲-起极限环 |
| gait-conditioned 奖励路由 | 跳跃窗口 (trigger on) 与站立窗口 (trigger off) **奖励路由隔离**: 跳跃奖励只进跳窗, 站立惩罚只进站立窗 (我们已是 phase-gated, 但 jump_height/height_progress 权重 75 太大, 站立惩罚太弱 — 需重平衡) |
| 站立命令课程 | 训练命令分布里把 trigger=0 站立期的时长占比拉高 (当前 ~50% 跳跃), 让站立成为主导模式 |

## 与我们的差异

- 文献多是四足/双足行走, 站立是"不走"的静止模式; 我们是轮腿, 站立还要保持倒立摆平衡 → 站立更难, 更需要显式稳定奖励。
- 文献 LPF 在动作层, 我们 VMC 变体动作是虚拟腿变量 (L0/θ0), 滤波位置需在关节力矩或虚拟腿参考层测试。
- 我们已有 phase-gated 奖励 (站立期跳跃奖励自动关闭), 缺的是**站立期惩罚权重太低** + **无 LPF** + **无站立显式稳定项**。

## 落地建议 (优先级 P3)

1. **加动作 LPF** (最便宜, 先试): `jump_srl.py` `step()` 里对策略输出做一阶低通 `a = a_prev + α(a_raw − a_prev)`, α≈0.3-0.5。
2. **站立期显式稳定惩罚**: 站立窗 (trigger≤0.5) 内 `ang_vel_xy` 提到 -0.5~-1.0, 或加"站立期 \|gyro\| 阶梯奖励" (如 exp(−\|gyro\|/0.3))。
3. **延长落地恢复窗口**: `landing_recovery` 门控从 landing_window 30 步改成"trigger 掉 0 后持续开 1-2s", 覆盖整个恢复期而不是 0.3s。
4. **命令分布重平衡**: 训练命令 resampling 让 trigger=0 占更高比例 (如 0/1 以 7:3 采样而非均匀), 强化站立模式。

**关联**: [[2026-08-16_jump_posture_knee_limit_vmc]] (P4/膝过伸), [[wheel_legged_lab]] (同构工程)
