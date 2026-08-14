# 参考文档: MPC × 强化学习 融合方法综述 (项目/论文)

**日期**: 2026-08-14
**来源**: 用户需求 — "帮我查询 MPC 与强化学习的融合项目, 和论文"
**关联**: [[ref_classic_control_survey]] (经典控制综述, LQR/MPC 基线依据);
[[../classic_lqr/INDEX.md]] [[../classic_mpc/INDEX.md]] (本项目已完成的 LQR + 线性 MPC 经典轨);
RL 侧 13 个任务 (walk_flat/walk_rough 等, [[05_retrain_all_13_tasks]])

**用途**: 论文 related work「经典 vs 学习」同口径对比的文献依据 + 融合任务轨的参考实现。
本项目已有 4 态线性 MPC (Hildreth QP) 与 LQR 独立任务轨 — 恰好可作为融合实验的"MPC 半边"。

---

## ★ 旗舰开源案例: AugMPC / IBRIDO (本地已有源码, 首选拆解对象)

**出处**: `github.com/AndrePatri/AugMPC` + `AndrePatri/ibrido-containers` (Andrea Patrizi, **IIT** 意大利技术研究院)。
本地已克隆在 `/home/robot/xlw/ibrido-containers/` (git remote 确认) — 可直接读源码。
论文: **RL-Augmented Model Predictive Control for Non-Gaited Legged and Hybrid Locomotion** (arXiv:2603.10878, IEEE RA-L)。
训练模型: HuggingFace `AndrePatri/AugMPCModels`。

**架构 (接触显式分层) — MPC×RL 融合的教科书式实现**:
```
高层 RL (SAC/PPO) ── 选择: 接触时序 + twist/导航指令
        │
        ▼
低层 MPC 集群 (800 并行刚体 MPC, 20Hz, ~1s 时域) ── 执行全身运动/力分配
```
- 把"接触时序"这一组合爆炸从 MPC 剥离交给 RL 学 → 产生**非步态(无节律)自适应步态**;
  单肢接触动作越过阈值即注入飞行相 (控脚离地/落地/飞行时长), 观测含原始高度图 → 非平地。
- **样本高效**: 800 并行 MPC + 全向量化仿真 (IsaacSim), 50+× real-time, SAC 1-10×10⁶ 步 (~6h)
  收敛, 比纯端到端盲 RL (>100×10⁶) 快一个数量级。
- **零域随机化 sim-to-real**, 覆盖 30-120kg 多形态 (含 120kg Centauro 轮足人形)。
- 工程栈: EigenIPC 共享内存 + `MPCHive` ControlClusterServer/Client (MPC 集群) + 世界接口
  (IsaacSim / xbot2 / 硬件) + `AugMPCTrainingEnvBase`。

**与本项目对照**: AugMPC 的"高层 RL 决策接触/指令 + 低层 MPC 执行"正是 xqrobotwl 融合轨可参照的
主流形态; 其 800 并行 MPC (JAX 之外用线程池/集群) 也回答了"MPC 太慢喂不动 RL"的工程难点。
更简单的替代: 本项目已有 numpy 向量化 Hildreth (0.05ms/步), 单环境 MPC 引导完全够用。

---

## 0. 为什么融合: 两种范式的互补性

| 维度 | MPC | RL |
|---|---|---|
| 模型 | 依赖 (显式动力学) | 免模型或学模型 |
| 约束处理 | ★ 显式, 硬约束/安全保证 | 弱 (软, 靠奖励惩罚) |
| 稳定性/保证 | ★ 可证明 (代价/末端约束) | 一般无 |
| 适应性 | 差 (模型失配崩) | ★ 强 (数据驱动, 泛化) |
| 计算 | 在线优化, 重 | 推理轻, 训练重 |
| 可解释性 | ★ 高 (成本/约束可读) | 低 (黑盒) |

两者几乎互补 → "MPC 给结构与安全, RL 给适应与鲁棒" 是 2024-2025 年融合综述的共识。

---

## A. 综述 (起点, 按图索骥)

1. **Synthesis of Model Predictive Control and Reinforcement Learning: Survey and Classification**
   — Reiter, Hoffmann, Reinhardt, Messerer, Baumgärtner, Sawant, Boedecker, Diehl, Gros.
   arXiv:2502.02133 (2025-02)。**最权威的分类综述**: 以 actor-critic RL 为框架, 系统分类 MPC 与 RL 的
   组合方式; 阐明两者同源于 MDP 但范式互补, 并给出各融合算法分类。**论文 related work 首选引用**。
2. **A Systematic Review and Taxonomy of RL-MPC Integration for Linear Systems**
   — Jalaeian-Farimani 等. arXiv:2604.21030。聚焦线性/线性化系统的 SLR, 多维分类学 (RL 角色、
   RL 算法类、MPC 形式、代价结构、应用域), 提炼共性问题: **计算负担 / 样本效率 / 鲁棒性 / 闭环保证**。
3. **Data Science and Model Predictive Control: A Survey of Recent Advances on Data-Driven MPC Algorithms**
   — Morato & Felix. Journal of Process Control, 2024。数据驱动 MPC 综述 (含 RL 方案), 强调"为复杂系统
   辨识可信模型"的挑战。

---

## B. 奠基: 学到的动力学模型 + MPC (模型基 RL 主线)

这条线先于"融合"概念存在, 是 MPC 出现在 RL 中的最早形态 (学模型 → 模型里做 MPC 规划):

1. **Neural Network Dynamics for Model-Based Deep RL with Model-Free Fine-Tuning (MBMF)**
   — Nagabandi, Kahn, Fearing, Levine. 2017。**首个** "NN 动力学 + MPC 采样子采样" 的样本高效范式:
   复杂运动任务比 TRPO 少 ~20× 数据; 再用 model-free 微调。局限: 收敛性能受模型精度限制。
2. **Deep RL in a Handful of Trials using Probabilistic Dynamics Models (PETS)**
   — Chua, Calandra, McAllister, Levine. NeurIPS 2018。**CEM-MPC + 概率集成动力学** 的经典基线:
   集成表征**认知不确定性** (分布外歧义), 高斯输出表征**偶然不确定性**; 少量轨迹达到 model-free 渐近性能。
3. **When to Trust Your Model: Model-Based Policy Optimization (MBPO)**
   — Janner, Fu, Zhang, Levine. NeurIPS 2019。给出**单调提升保证** (模型偏差 C 分解为泛化误差+分布漂移),
   提出 **branched rollout** (从真实数据分布状态出发的短模型回放), 成为"安全用模型"的标准配方。
4. **Information Theoretic MPC for Model-Based RL (MPPI)**
   — Williams 等. ICRA 2017。信息论 MPC, 无导数采样规划; 后续大量与学习结合。
5. **World Models / Dreamer 系** — Ha & Schmidhuber 2018 (NeurIPS); Hafner 等 (Dreamer 2020)。
   潜空间世界模型 + 模型内 MPC/规划, 视觉任务样本高效主流。

> 与本项目关系: 本项目 MPC 用的正是**黑箱系统辨识模型** (`logs/classic/mpc_plant_bb.npz`, OLS 拟合
> A_d/B_d) — 是这条线"单步线性化"特例; 融合方向 B 的"概率集成 + MPC"可改善其 P4 地形鲁棒。

---

## C. 可微 MPC / MPC 作为策略类 (端到端学习 MPC 组件)

1. **Differentiable MPC for End-to-end Planning and Control**
   — Amos, Jimenez, Sacks, Boots, Kolter. NeurIPS 2018。**奠基作**: 把 MPC 当作**可微策略类**用于连续
   控制 RL; 通过凸近似的 **KKT 条件在不动点解析求导** (而非 unroll 优化过程), 端到端学习代价与动力学。
   在 pendulum/cartpole 上比通用 NN 策略数据效率显著更高, 优于 vanilla 系统辨识。
   - 代码: `github.com/bamos/differentiable-mpc` + 独立库 `locuslab/mpc.pytorch` (PyTorch)。
2. **Actor-Critic Model Predictive Control**
   — Romero, Song, Scaramuzza。**把可微 MPC 嵌入 actor-critic RL 框架**: 结合 model-free RL 性能与
   MPC 的鲁棒重规划。对"纯 RL 抖动 + 纯 MPC 脆"是直接示范。
3. **MPCritic: A Plug-and-Play MPC Architecture for Reinforcement Learning**
   — Lawrence, Banker, Mesbah. 2025。ML 友好的 MPC 架构, 与 RL 工具链即插即用。
4. **Local-Global Learning of Interpretable Control Policies: The Interface between MPC and RL**
   — Banker, Lawrence, Mesbah。局部决策器近似满足全局 Bellman 方程的 local-global 范式 (可解释策略)。

---

## D. MPC 引导 RL 训练 (物理先验当奖励/参考)

核心思想: MPC 在线生成高质量参考轨迹 → 替代手写奖励 / 提升样本效率与收敛。

1. **Online Parallel MPC-Guided Reinforcement Learning Framework for Legged Locomotion** (IEEE 2025)
   - 物理先验 MPC 在线生成参考轨迹引导策略学习; 关键工程: **JAX + GPU 并行 MPC**, 集成 NVIDIA Isaac Lab;
   四足验证。解决了"MPC 太慢没法批量喂给 RL"的痛点。
2. **eGAIT: Multi-Skilled Policy for Energy-Efficient Gait Transitions** (IEEE 2025)
   - MPC 步态生成器产速度优化轨迹 → PPO + AMP 式奖励模仿; 分层 RL 统一多技能; Unitree Go1 验证,
   节能/速度跟踪/稳定提升。
3. **CrossLoco** (2023) — 人类运动驱动 RL, 无监督; 非 MPC 但同"参考引导"族。
4. **Residual RL** (参考, 经典基 + RL 残差, 如 Silver 等 2018 "Residual RL for Robot Control")
   - 经典控制器作底座策略, RL 学残差修正 — **与本项目 LQR/MPC 基线最顺滑的融合入口**:
     已证明降低 RL 探索难度、保留经典稳定语义。

---

## E. 分层/层级融合 (RL 高层决策 ↔ MPC 低层执行, 或反之)

1. **PIP-Loco: A Proprioceptive Infinite Horizon Planning Framework** (ICRA 2025)
   - 训练期专家策略 + 内部模型 (速度估计 + Dreamer 模块) 协同学习; 部署期 Dreamer 模块**以 MPC 方式解
     无限时域优化**过滤违约束动作/速度指令。Unitree Go1 多地形验证。
2. **VIP-Loco: A Visually Guided Infinite Horizon Planning Framework** (Stoch Lab, 2025)
   - PIP-Loco 视觉版; 在**四足 Go1 + 双足 Cassie + 轮足双足 TronA1-W** 上验证 (含**轮足**!),
   难度地形收益最大, 轮足双足收益最显著 — **与本项目平台类型最贴近的一作**。
3. **PlanNetX: Learning an Efficient Neural Network Planner from MPC** (Hoffmann 等, 2024)
   - 把 MPC 蒸馏成 NN planner (教师-学生), 线上用轻量 NN 逼近 MPC。
4. **Virginia Tech 多智能体分层控制** (硕士论文, 2025) — 高层 **DNMPC (ADMM, 5 Hz)** 生成速度指令,
   低层 **端到端 PPO (250 Hz)** 跟踪; Unitree A1 多智能体导航验证。
5. **Hierarchical RL with Low-Level MPC for Multi-Agent Control** — Studt & Schildbach。
   高层 RL 战术决策 + 低层 MPC 执行。
6. **MPC + 预测 RL 滚动 + 末端 Q 函数** (arXiv 2307.07752 一带) — 用 NN Q 函数作 MPC 尾部代价, 缓解
   长时域指数复杂度; A1 上短时域即稳, 无需预训练。

---

## F. 安全屏蔽: MPC 给 RL 当安全层 (Safe RL)

1. **Safe RL via Shielding** — Alshiekh 等. 2018。奠基: 盾牌限制 RL 动作满足安全规格。
2. **NMPC safety filter for multi-agent navigation** (arXiv 2312.12861)
   - RL 动作过 NMPC 安全滤波: `min ‖u−u_RL‖ s.t. 动力学+约束`, 且 RL 对偏离 MPC 安全动作**加惩罚**,
     训练策略"不依赖安全网"。→ 收敛时 MPC 介入率自然降低。
3. **Safe RL via adaptive robust model predictive shielding** (Computers & Chemical Engineering, 2025)
   - 离线鲁棒 NMPC 后备策略 + 在线自适应安全参数进观测空间; 并行 min-max 鲁棒 rollout (4 条), 比
     蒙特卡洛 1000 条才安全高效得多。
4. **CBF-QP 安全盾** (Frontiers in Neurorobotics, 2025)
   - 每步解 QP `u*=argmin‖u−u_nom‖² s.t. CBF 不变集约束`, 违约束时把 RL 动作投影到最近安全动作;
     在线训练期也启用避免收集危险数据; 返回活跃约束的 Lagrange 乘子 (安全度信号)。
5. **Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions**
   - 采样 MPC + 学习 CBF, resampling-reweighting 滤违约束样本, 证明方差有界。
6. **Vision-driven River Following of UAV via Safe RL using Semantic Dynamics Model (CADE)**
   - 代价估计器 + 语义动力学模型, cost-planning 安全滤波器实时动作叠加。

> 与本项目关系: 两轮足是**静不稳定+欠驱动**, 硬安全约束 (倾角/轮速/关节限位) 恰好是 MPC 强项 —
> 「RL 提案动作 → MPC 安全层投影」思路可直接移植到 walk_rough, 缓解 P4 0% 问题。

---

## G. 轮足机器人专项 (与平台最相关)

1. **FLORES 轮腿机器人 HIM 框架** — 经典 **内模控制 (IMC) + PPO**: 仅用本体感知估计地面摩擦/地形高程
   等环境扰动, RL 补偿内模误差。**与本项目 (轮足+PPO) 平台与问题几乎同构**。
2. **L1-MPC** — 把 L1 自适应控制融入 MPC 公式, 对未建模动力学鲁棒。
3. **NeuroMHE** — 循环网络 + 移动时域估计推断潜在扰动。
4. **MPC + 加权多任务 WBC (WM-WBC)** — IEEE RA-L 2025, 轮足双足分层优化 (轮动力学 + 质心动力学,
   滚动约束), 处理非最小相位与欠驱动; 模型基路线。
5. **Whleaper** — 轮足双足用 PPO 走+跳; 纯 RL 对照。
6. **IJRCS 2023 轮足双足综述** — 编目"RL+MPC 效率融合 / 挑战 MPC 假设的优化法 / RL 调 MPC 参数"。
7. **自适应动态最优平衡 (Applied Mathematical Modelling, 2025)** — RL/自适应动态规划启发的轮足不平地形
   最优平衡。
8. **Neural Approximation-based MPC of Non-holonomic Wheel-legged Robots** — RBF 神经网络估计不确定
   动力学+外扰, 模型预测跟踪控制。
9. **Diffusion-MPC (Harvard SEAS)** — 四足上 MPC + 生成扩散模型实时行为调节 + 物理/安全约束。
10. **Direct Benchmarking: MPC vs RL for Legged Locomotion in MuJoCo** (IEEE Access, 2025)
    - 首次直接对比 Go1 平地直行: RL 抗扰/能效优但地形泛化弱; MPC 大扰动恢复强。**正是本项目
      "LQR/MPC vs RL 同口径对比"论文表的目标形态**。

---

## H. 开源项目/框架 (可直接参考)

| 项目 | 内容 | 相关章节 |
|---|---|---|
| **AndrePatri/AugMPC + ibrido-containers** | ★ **RL×MPC 融合旗舰** (IIT): 高层 RL 接触/指令 + 低层 800 并行 MPC; arXiv:2603.10878; **本地已克隆** `/home/robot/xlw/` | ★顶层 |
| **FilippoAiraldi/mpc-reinforcement-learning (`mpcrl`)** | TU Delft, RL 在线调 MPC 代价参数 (Q-learning/DPG/贝叶斯优化), ~700★, MIT | C |
| **NPLawrence/RL-MPC** | 值函数增强 MPC: RL Q 函数当 MPC 末端代价 | E |
| **google-deepmind/mujoco_mpc (MJPC)** | MuJoCo 实时预测控制: iLQG / 梯度下降 / Predictive Sampling 规划器; C++ 核心 + Python API; Apache-2.0; 论文 arXiv:2212.00541 | B/D |
| **bamos/differentiable-mpc** + **locuslab/mpc.pytorch** | Amos 2018 的可微 MPC 实现 | C |
| **do-mpc / acados / CasADi** | 通用 (N)MPC 工具箱: 鲁棒 MPC+MHE / 嵌入式实时 NMPC / 符号求导后端 | 工程备选 |
| **NVIDIA Isaac Lab + JAX MPC** | GPU 并行 MPC 引导 RL (IEEE 2025 用) | D |
| **saucesaft/differential_policies** | JAX/MJX 可微仿真直接训练策略 (SHAC), monkey-patch 接触软激活 | C |
| **SIRI611/motion-imitation** | 参考运动模仿 RL 基线 | D |
| **github.com/bamos/* 系 / zhenghongyu1986 / passion4energy / weiqiao 的 differentiable-mpc forks** | 复现与教学 | C |

> 工程上注意: MJPC 侧重**经典规划器**, 非融合; 融合工程主要靠把 MPC 求解器做成**可微/可并行**模块
> (JAX) 再塞进 RL 循环 — 本项目已有 numpy 向量化 Hildreth QP, 未来可 JAX 化做并行 MPC 引导。

---

## I. 对本项目 (xqrobotwl) 的落地映射

现状: 已完成 LQR + 线性 MPC 独立轨 (P1-P3 100%, P4 rough 0% 遗留), 13 个 RL 任务重训中。
融合候选 (按工程成本排序):

| 优先级 | 方向 | 文献依据 | 预期收益 |
|---|---|---|---|
| ★★★ | **MPC 安全层 (Safe RL shield)** | F2/F3/F4 | 解 P4 rough 0%: RL 提案 → MPC 投影到安全 (倾角/轮速/关节限位), 且 RL 加偏离惩罚 |
| ★★★ | **Residual RL: MPC 底座 + RL 残差** | D4 | 在 MPC 轮速命令上加 RL 残差, 保留经典稳定语义, 提升地形泛化 |
| ★★☆ | **MPC 引导 RL 参考轨迹** | D1/D2 | 手写奖励换成 MPC 轨迹奖励, 收敛更稳 (但需 GPU 并行 MPC) |
| ★★☆ | **可微 MPC / 代价学习** | C1/C2 | 端到端学 MPC 代价 (本项目 Q 权重), 免手动调参 |
| ★☆☆ | 世界模型 + 潜空间 MPC | B5 | 长时域/视觉地形, 工程量大 |
| 对照 | **MPC vs RL 同口径基准** | G10, 论文相关表 | 直接产出论文「经典 vs 学习」对比表 |

> 论文相关 (related work) 章节最省力引用: A1 (综述) + C1 (可微 MPC) + E2 (轮足 VIP-Loco) + G1 (FLORES HIM) + G10 (基准)。

---

## 关键 arXiv/出处速查

- arXiv:2502.02133 — MPC-RL 融合综述 (Reiter 2025)
- arXiv:2604.21030 — RL-MPC 线性系统系统综述
- arXiv:1810.13400 — Differentiable MPC (Amos 2018, NeurIPS)
- arXiv:2212.00541 — Predictive Sampling / MuJoCo MPC
- arXiv:2409.09441 — PIP-Loco (ICRA 2025)
- arXiv:2307.07752 — MPC + 预测 RL 滚动 + 末端 Q 函数
- arXiv:2312.12861 — NMPC safety filter 多智能体
- PETS (Chua 2018, NeurIPS) / MBPO (Janner 2019, NeurIPS) / MBMF (Nagabandi 2017)
