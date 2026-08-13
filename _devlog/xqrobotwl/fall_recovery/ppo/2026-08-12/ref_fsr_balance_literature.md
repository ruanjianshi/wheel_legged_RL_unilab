# 参考文档: 跌倒恢复 + 两轮倒立摆平衡文献调研 (v8.x 通宵迭代依据)

**日期**: 2026-08-12
**来源**: CLAUDE.md §7.8 "开发新姿态/优化任务效果, 需基于问题查询论文/开源项目学习参考, 并写参考文档"
**关联**: v8 桥式局部最优失败 → 文献调研; v8.1 rise_vel 直立门控修复

---

## 1. FTSR 论文 (本项目跌倒恢复的实现依据)

**论文**: Hou, Yu, et al. "Robust Fall Recovery for Armless Bipedal-Wheeled Robots via Force-Guided Learning." arXiv:2606.14270 (IEEE RA-L 2026). 项目页: https://2350575870.github.io/force-guided.github.io/

### 关键方法 (与本地实现对应)

| FTSR 组件 | 本地实现 | 状态 |
|---|---|---|
| **CMDP 力约束** (非启发式力课程): 外助力 F∝(1−e^{−μ(h_cmd−h)}), 作为**可优化约束代价**(非奖励), 约束限 d=0 → 策略渐进降低对助力依赖 | CPO 双约束 C1=F/C2=T, beta 罚函数法, d=0 | ✅ 已实现 |
| **高度分阶段奖励** (上部转正 → 站立 → 行走), 按批次高度统计动态切换目标 | ru (h_cmd1=0.32) → rs (h_cmd2=0.52), stage_fraction 切换 | ✅ 已实现 (无行走阶段) |
| Teacher-student 蒸馏 (特权信息 → 本体感知) | 未实现 (纯学生, 仅仿真无 sim2real 需求) | — |

### ★ 对本地 v8 失败的启示

1. **"策略被引导渐进降低对助力依赖"** — FTSR 的力是**约束**, 整个训练过程 CPO 都在压助力使用,
   不是只在力衰减末期才适应。本地 v8.1 观察到: 恢复在 iter 2200-2300 成形 (力强时) 后,
   随力撤 (force_end_iters=3000) **回退** → 提示策略依赖了力。需检查:
   - CPO 约束系数 (beta_init=0.001, beta_max=0.1, beta_growth=1.0001) 是否太弱, 压不动助力依赖
   - force_end 线性衰减 (sat(1−steps/force_end)) 是否形成"悬崖"而非渐进
2. **"过度约束力引导 (惩罚乘子过大) 会降低鲁棒性"** — 若 beta 过大也会崩。
   折中: 约束强度要够让策略自举, 又不能压死。
3. **消融: 去掉阶段奖励或力约束会严重降恢复率** — 本地里程碑阶梯 (rise/recover_complete)
   与 CPO 力约束都是必需, 不可删。

### 后续可尝试 (基于论文)

- 若 v8.1@4000 仍未达标: 调整力约束策略 — 提高 beta_init (更早压助力依赖),
  或 force_end 后置到 5000 并放缓衰减, 让适应窗口更长
- 力撤回归是 FTSR 已解决的问题, 方向是"约束压依赖 + 阶段奖励推高度"共同作用

## 2. 两轮倒立摆站立平衡文献 (v8 锚点/放松约束设计依据)

核心结论: **平衡必须靠轮子持续微调** (前后来回/差速/yaw 修正), 任何"罚瞬时轮速/差速/角速度"
的约束都会削弱平衡自由度 → 站立时长下降。这正是 v7 站立塌到 0.63-0.81s 的根因,
v8 的"放松瞬时约束 + 锚点管净位移/净旋转"方向与之吻合。

| 参考 | 要点 | 对本地启示 |
|---|---|---|
| Gain-Scheduled Control, Ascento (IEEE 2026, [doc/11565592](https://ieeexplore.ieee.org/document/11565592)) | 增益调度降 pitch 误差 8.5% / 位置跟踪误差 16.8% | 站立期可按"已站稳"增益切换 (近似 anchor 门控) |
| LPV/gain-scheduling 15+ 控制策略对比 (GitHub [Manas-arumalla](https://github.com/Manas-arumalla/Control-strategies-for-two-wheeled-inverted-pendulum), MuJoCo 验证) | LQR/MPC/H∞/SMC 等对倒立摆有效 | 轮子平衡本质是连续微调, 非静态 |
| Wheel-synchronization / motor mismatch 补偿 (Robotics 2026) | 主从轮同步环补偿电机失配, 不影响全局平衡 | 差速微调是"手段", 净转圈才是"问题" — 支持 v8 yaw 锚点 |
| Reaction-wheel PID (Sunway) | PID 增益调参消振荡/超调 | 平衡 = 持续扭矩调节 |

**设计确认**: v8/v8.1 的"放开平衡微调自由 (wheel_symmetry τ/12, no_yaw/stand_still 放松) +
锚点管净位移/净旋转 (stand_anchor: σ_xy 0.25m / σ_yaw 0.35rad)" 与文献一致 —
平衡给自由度, 漂移/转圈管净量。

## 3. 综合判断

- 恢复期: FTSR 力约束 + 阶段奖励方向正确, 问题是"力撤适应", 优先试更长的适应窗口 /
  更强约束压依赖
- 站立期: v8 锚点方向正确 (文献支持), 待恢复成形后验证站立时长/漂移是否达标
- 若 8 点前时间不足: 优先保住"恢复率≥80%"主判据, 站立/漂移作为后续增量

## Sources

- https://arxiv.org/abs/2606.14270 (FTSR, RA-L 2026)
- https://2350575870.github.io/force-guided.github.io/ (FTSR 项目页)
- https://ieeexplore.ieee.org/document/11565592 (Ascento gain-scheduled balance)
- https://github.com/Manas-arumalla/Control-strategies-for-two-wheeled-inverted-pendulum (15+ 策略对比)
