# 参考文档: 两轮足机器人经典控制算法综述 (新增经典算法基线的文献依据)

**日期**: 2026-08-13
**来源**: 用户需求 — "查询经典算法, 常用的两轮足机器人经典算法" + "整理成文献笔记" + "给强化学习项目新增几个经典算法控制"
**关联**: [[ref_fsr_balance_literature]] (倒立摆平衡文献, v8 站立依据); 项目已有经典控制先例:
`tools/xqrobotwl/single_leg_lqr.py` (独轮平衡 LQR/极点配置, 数值线性化+闭环验证),
`tools/xqrobotwl/single_leg_classic_control.py` (独轮车 OLEBOT/unicycle LQR),
`src/.../single_leg_move.py` (LQR 参考引导 RL), `src/.../vmc.py` + `jump_vmc.py` (虚拟模型控制跳跃)

---

## 0. 核心难点 → 决定算法分层

两轮足机器人是**欠驱动、非最小相位、静不稳定**系统 (仅两轮着地, 其余全凭主动平衡),
倒立摆模型是"动态心脏"。因此**平衡控制是第一核心问题**; 加上腿部自由度还要解决
**动态技能** (跳跃/跌倒恢复/越障) 与**状态估计**。经典算法按任务分层:

| 层 | 经典算法 | 代表平台 |
|---|---|---|
| 平衡 (模型基) | 串级PID · LQR/LQG · 极点配置 · 增益调度/LPV · 滑模 · 反步 · H∞ · 自适应 | 平衡车/Segway, I-PENTAR |
| 预测/最优 | MPC / NMPC (EPSAC) | Ascento 系, WLR-3P, DIABLO |
| 全身协调 | WBC(全身控制) · VMC(虚拟模型) · ZMP | Ascento, Ollie |
| 学习基 | PPO · SAC · DDPG/TD3 · MoE · Residual Policy · sim-to-real | 松灵/宇树/瑞士-Mile 系 |

## 1. 平衡控制 (模型基经典)

- **串级 PID** — 最基础经典法, 典型"内环角速度 + 外环倾角"(平衡车范式), 腿长 PID 后串加速度环抗扰。
  简单, 但对复杂地形适应性差、依赖手动调参。
- **LQR(线性二次型)** — **两轮足平衡最常用经典算法**。倒立摆平衡位线性化 → 解 Riccati 得最优反馈。
  实测 0.2 rad 扰动 ~2s 回正并回原点。常搭 **LQG**(Kalman 观测器)、**极点配置**、**自适应 LQR**、
  **NN 动力学参数辨识**补偿模型误差。局限: 依赖模型精度, 机身高度等参数变化需手动重调。
- **MPC / NMPC** — 显式处理约束 (轮速/倾角/力矩上限), 扰动下可比 LQR **省 ~50% 轮力矩**;
  代价是计算量大, 实时受限时常以 LQR 为降级后备。
- **滑模 SMC / 反步 / H∞ / LPV** — 鲁棒类: 模型不确定时表现好; 滑模有抖振问题, 常与模糊/NN 混合;
  LPV/增益调度覆盖参数变化 (机身高度变化)。
- **综述共识**: 模型基稳定保证 + 数据驱动适应性的**混合控制**是主流趋势。

## 2. 全身协调 (Ascento 一系)

- **WBC + LQR 任务** — ETH Ascento 2: 分层 WBC 用全刚体动力学, 把 **LQR 反馈律作为运动学任务嵌进
  WBC 层次**处理非最小相位平衡; 四杆腿部运动学开环 + 闭环约束力重闭; **弯道防倾翻用 ZMP 调躯干前倾**。
- **VMC(虚拟模型控制)** — 广泛用于腿控: 策略输出虚拟腿角/腿长/轮速 → VMC 解算 4 腿关节 + 2 轮力矩。
  **本项目 jump 的 VMC 变体即此系** (已有 `vmc.py`)。
- **空中姿态**: 离地检测 (支撑力 FN<20N) → 只保腿部姿态竖直, 平稳落地。

## 3. 动力学模型 (控制/规划的基础)

- **倒立摆族**: 单级 SIP, 二级 DIP (俯仰), 轮式 **W-LIP / W-SLIP**。
- **跳跃虚拟模型**: **NW-SLIP** (非线性轮式 SLIP, 较线性弹簧跳高 +3.4×), **DMLSM** (双质量线性弹簧)。
- **SLIP 原点**: Raibert 跳跃机器人 → 连续跳跃规划的经典简化模型。

## 4. 状态估计

- **互补滤波** — 陀螺积分 (低噪漂移) ⊕ 加速度计倾角 (噪声但绝对), θ̂=c·θ̂+ωdt+(1-c)·θ_accel;
  **互补 Kalman 滤波**变体加陀螺偏置估计。低算力平台常用, 有实现实测优于 Kalman。
- **EKF** — 结合动力学模型; 高 yaw 率机动需 **3D 动力学 EKF** (平面模型假设失效)。
- **α-β 滤波 / 编码器+里程计融合** — 里程计受轮滑影响; **自适应 Kalman** (在线调 Q/R) 用于防滑。

## 5. 动态技能: 跳跃 / 跌倒恢复

- **轨迹优化 (TO)**: 动态模型 + 约束生成物理一致跳跃轨迹 (起跳/腾空升/腾空降/触地四阶段),
  QP/二分/极小极大求解, 配阻抗/PD 跟踪 + 跳跃状态观测器。综述结论: TO 与学习法两条主线,
  未来难点在重载、sim-to-real、技能组合。
- **跌倒恢复**: Ascento 多阶段起身 (收腿 → 受控起升 → 施加力矩), 覆盖 4 种卧姿中的 3 种 + 受控摔倒。
  **这是本项目 fall_recovery 的模型基对标** — 本项目用学习法 (CPO+FTSR) 已达仰卧 30%/俯卧 90%+。

## 6. 学习基 (近年主流, 本项目所在)

- **无模型 DRL**: PPO (本项目 walk/jump), CPO (fall_recovery 力引导约束), SAC/DDPG/TD3。
- **架构演进**: MoE 解耦滚-腿模式梯度冲突; Residual Policy + trust-region (地形突变鲁棒, 成功率 96.7%);
  asymmetric actor-critic + 特权学习 (340kg 重载, 3.8 m/s); HIM 混合内模。
- **训练范式**: 课程/分阶段 (平平衡→移动→跳跃→带障), 力引导辅助, sim-to-real 域随机化。
  开源参考: **Wheel-Legged-Lab** (Isaac Lab + RSL-RL PPO + VMC)。

## 7. 经典平台对照

| 平台 | 结构 | 算法 |
|---|---|---|
| 平衡车 / Segway | 两轮倒立摆 | 串级 PID, 极点配置, LQR |
| **Ascento 1/2** (ETH) | 两轮双腿四杆 | WBC + LQR + ZMP, 跳跃 TO, 多阶段跌倒恢复 |
| Boston Dynamics Handle | 两轮双腿 | 模型基 + 学习基混合 |
| OLEBOT / DIABLO / WLR-3P | 两轮双腿 | W-LIP/W-SLIP, MPC 跳跃 |
| 松灵/宇树 轮腿系 | 两轮双腿 | RL (PPO/SAC) 为主, VMC 辅助 |

## 8. 与本项目 xqrobotwl 对照

| 任务 | 本项目算法 | 可对照的经典算法 |
|---|---|---|
| walk_flat / toe_walk / rough | PPO | LQR 平衡 + MPC 速度追踪 (指令追踪基线) |
| jump | PPO / PPO+VMC / SRL+VMC | SLIP 虚拟模型 + 轨迹优化跳跃 |
| backflip | PPO + 确定性 FSM 前馈 | 开环力矩前馈 (已部分实现) |
| single_leg | PPO + LQR 参考引导 (已实现) | LQR/极点配置 (已有 `single_leg_lqr.py`) |
| fall_recovery | CPO + FTSR 力引导 | Ascento 多阶段起身 (模型基对标) |
| stairs | NP3O | MPC 全身跟踪 |

**现状**: 经典控制在本项目已有雏形 — `single_leg_lqr.py` 独轮平衡 LQR + VMC 跳跃 +
jump 的 FSM 前馈。缺口在**把经典控制器做成可对比基线** (统一进 `_devlog/assess` 评估,
与 RL policy 用同一套指标) — 见推荐方案。

## 9. ★ 推荐落地算法 (新增经典控制基线)

选型原则: ① 贴合两轮足本质 (倒立摆) ② 复用已有先例模板 (`single_leg_lqr.py` 数值线性化+闭环验证)
③ 能进 `_devlog/assess` 统一评估 (与 RL 同指标对比) ④ 对论文/毕业论文 related work 有价值。

| 优先级 | 算法 | 对应任务/场景 | 价值 | 工作量 |
|---|---|---|---|---|
| **P0** | **LQR 平衡控制器** (两轮倒立摆) | fall_recovery 站立期 / walk 无指令微动平衡 | 经典标杆, 模板现成 | 低 |
| **P0** | **串级 PID** (倾角PD + 轮速PI) | 同上 (最简参照) | 对比表里"经典 vs 学习"极简条目 | 极低 |
| **P1** | **线性 MPC** (带约束) | walk_flat 速度/位姿追踪 + 抗扰 | 约束处理强, 省力矩 | 中 |
| **P2** | **VMC 腿控** (扩展已有 jump VMC) | walk_flat 腿部协调 | 已有基础 | 中 |
| **P2** | **SLIP 虚拟模型跳跃** | jump 高度规划 | 论文对照亮点 | 高 |

**统一评估路径**: 经典控制器包成 `policy(obs) → action` 的 callable, 挂进 `_devlog/assess/runner.py`
的 `load_policy` 分支 (RL 是 `load_actor(ckpt)`, 经典是 `load_classic(kind, gains)`), 复用
`run_episodes` + 指标 + 达标判定 → 与 RL 同口径对比 (恢复率/漂移/追踪误差/站立时长等)。

## 参考文献

- [Control Strategies for Two-Wheeled Self-Balancing Robotic Systems: A Comprehensive Review (MDPI)](https://www.mdpi.com/2218-6581/14/8/101)
- [Control-strategies-for-two-wheeled-inverted-pendulum: 15+ 算法对比 LQR/MPC/H∞/SMC/反步/LPV/自适应 (MuJoCo 验证, GitHub)](https://github.com/Manas-arumalla/Control-strategies-for-two-wheeled-inverted-pendulum)
- [Ascento: A Two-Wheeled Jumping Robot (ICRA 2019, arXiv)](https://ar5iv.labs.arxiv.org/html/2005.11435)
- [LQR-Assisted Whole-Body Control of a Wheeled Bipedal Robot With Kinematic Loops (IEEE RA-L 2020)](https://www.x-mol.com/paper/1409467758439878656)
- [Jump Planning and Airborne Attitude Control of Bipedal Wheel-Legged Robots (IEEE, W-LIP/W-SLIP)](https://ieeexplore.ieee.org/abstract/document/11076174)
- [Jump Control Based on Nonlinear Wheel-SLIP Model (NW-SLIP, HIT, Biomimetics 2025)](https://scholar.hit.edu.cn/en/publications/jump-control-based-on-nonlinear-wheel-spring-loaded-inverted-pend/)
- [Jumping in legged robots: A review (Robotics and Autonomous Systems, 2026)](https://www.sciencedirect.com/science/article/abs/pii/S0921889026001077)
- [Attitude estimation of a high-yaw-rate MIP: EKF vs Complementary Filter (ACC 2018)](https://ieeexplore.ieee.org/document/8431624)
- [Adaptive multi-mode locomotion for bipedal wheel-legged robots via sparse MoE DRL (Frontiers 2026)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1788395/full)
- [Residual Policy Optimization With Trust Region Constraints (IEEE TASE 2025)](https://www.mendeley.com/catalogue/56649568-b0dd-3f1f-a0a9-8fdd5b1d71cd/)
- [Wheel-Legged-Lab: Isaac Lab RL 开源实现](https://github.com/zyicome/Wheel-Legged-Lab)
- [轮腿式平衡机器人控制 (信息与控制)](https://xk.sia.cn/cn/article/doi/10.13976/j.cnki.xk.2023.2533)
- [双轮足机器人控制方法总结 (bilibili)](https://www.bilibili.com/opus/1030478481038770195)
- [Learning-Based Balance Control of Wheel-Legged Robots (IEEE)](https://ieeexplore.ieee.org/document/9497675)
