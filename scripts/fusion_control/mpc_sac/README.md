# MPC×SAC 融合控制框架

**高层 SAC 决策(残差校正)+ 低层线性 MPC 执行** — AugMPC (IBRIDO) 架构移植到两轮足 xqrobotwl。
独立自包含任务轨: 自带 MPC (`mpc/`) + 自带 SAC (`sac/`), 只读复用共享基础设施 `common/` (§3.2)。

## 架构总览

```
  传感器 (θ,θ̇,v,ω_z,base_z) + 期望命令 (vx,vyaw,height)
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 高层 SAC 策略                │  obs 11D → 残差校正 a∈[-1,1] (3D)
   │ GaussianActor + TwinQ       │  (flat; rough 9D→2D)
   │ 温度自学习 + obs 归一化      │
   └──────────────┬──────────────┘
                  │ a (残差校正, a≈0 表示"不需要校正")
                  ▼
        cmd = des + res_scale·a      ← ★ 残差式融合
                  │
                  ▼
   ┌─────────────────────────────┐
   │ 低层线性 MPC (冻结执行器)     │  轮速命令 + 腿目标 + 偏航差分
   │ Hildreth QP + 5态积分增广    │  → a8 (RL 空间 8D)
   └──────────────┬──────────────┘
                  ▼
             env.step(a8)
```

## 分工: 为什么这么分

| 层 | 角色 | 承担 | 不承担 |
|---|---|---|---|
| **MPC (低层)** | 稳定骨架 | 倒立摆平衡、轮速命令、腿目标、偏航差分、显式约束 (倾角/轮速/关节) | 长时域策略、地形高维特征 |
| **SAC (高层)** | 适应性 | 学"何时校正指令"、补偿 MPC 模型误差、适应地形/扰动 | 重新学平衡 (MPC 已做) |

互补: MPC 给**结构**(约束、稳定、可解释), RL 给**适应**(数据驱动)。这正是融合综述
(arXiv:2502.02133) 与 AugMPC (arXiv:2603.10878) 的共识。

## 残差式融合 (★ 关键设计)

初版让 RL **直接输出命令**, 结果策略要"重新学" MPC 已经会的速度跟踪, policy+MPC 闭环震荡。
改为**残差式**:

```
cmd = des + res_scale · a        a ∈ [-1,1] 为 SAC 输出
res_scale: vx ±0.2, vyaw ±0.05, height ±0.02 (围绕期望的小校正)
```

- **a≈0 → cmd≈des → MPC 基线直接跟踪** → 初始化即稳定 (可回退到经典 MPC)
- RL 只在"需要校正"时输出非零 (补偿滞后/模型误差/地形)
- 校正带小 → 不可能失控 (自旋/过冲被界住)

## 训练 (低层 MPC 冻结)

```
train_mpc_sac.py
  └─ MpcSacMetaEnv (64 env 向量化, 每 env 独立 MPC 内部状态)
       ├─ reset: 站姿复位 + 1.5s 站姿预热 (MPC 起步需站稳)
       ├─ step(a):  denorm(a)→cmd → 每 env MPC.act→a8 → env.step
       │            → reward + obs + done (auto-reset)
       └─ SAC:  replay buffer → 软更新 (GaussianActor + TwinQ + α 自学习)
```

**奖励** (对 MPC 执行后的轨迹):
```
r = w_alive·(1−done)
  + w_vx·exp(−(v−des_vx)²/2σ²)        # 速度跟踪
  + w_vyaw·exp(−(ω_z−des_vyaw)²/2σ²)  # 偏航跟踪
  + w_h·exp(−(base_z−des_h)²/2σ²)     # 高度跟踪 (平地)
  + w_θ·θ² + w_ω·ω_z²                 # 直立 + 防自旋
  + w_corr·‖a‖²                        # 残差校正惩罚 (a≈0 除非真需要)
```

## 关键文件

```
scripts/fusion_control/mpc_sac/
├── mpc/            # 分支自带线性 MPC (移植经典轨, 数值一致)
│   ├── qp.py          Hildreth QP (build_mpc_matrices/solve_qp/dlqr_riccati)
│   ├── dynamics.py    轮速命令模型 + 黑箱模型加载
│   └── controller.py  低层 MpcController (sagittal QP + act 骨架)
├── sac/            # 紧凑 SAC 高层策略
│   ├── networks.py    GaussianActor + TwinQ
│   ├── replay.py      回放缓冲
│   └── trainer.py     SAC 更新 + obs 归一化 + 温度自学习
├── config.py       MpcSacConfig (conf/fusion_control/mpc_sac 自包含)
├── env.py          build_env/read_sensors 向量化 (自包含 conf)
├── obs.py          高层 obs / 残差 denorm / 期望采样 / reward
├── controller.py   MpcSacController (SAC→cmd→MPC→a8, 推理)
├── meta_env.py     MpcSacMetaEnv (训练向量化 env, 站姿预热, auto-reset)
├── runner.py       run_episode (record 对齐 common.metrics)
├── metrics.py      融合指标 (约束违犯率/残差幅度/求解耗时)
├── train_mpc_sac.py   SAC 训练入口
├── eval_mpc_sac.py    批量评估 CLI
└── balance_mpc_sac.py 单次运行 + 渲染 CLI
conf/fusion_control/mpc_sac/   robot/commands/config/task 自包含
```

## 验收 (平地 F1b, 5 ep)

| 指标 | 融合 | 阈值 | 判定 |
|---|---|---|---|
| 存活率 (15s) | **100%** | ≥95% | ✅ |
| gyro_rms | 0.136 | <1 | ✅ |
| vx_rmse (vx=0.4) | **0.039** | <0.1 | ✅ |
| height_err (平地地形) | **0.034** | <0.05 | ✅ |
| θ_max 违犯率 | 0 | =0 | ✅ |
| 低层求解 | 0.14ms | <10ms | ✅ |

对比: 纯 MPC P2 vx_rmse 0.087 → 融合 0.039 (残差学习改善速度跟踪)。

## 运行

```
# 训练 (64 env, 3000 iter)
uv run python scripts/fusion_control/mpc_sac/train_mpc_sac.py --task walk_flat

# 评估
uv run python scripts/fusion_control/mpc_sac/eval_mpc_sac.py --task walk_flat \
    --checkpoint logs/fusion_control/mpc_sac/walk_flat/<run>/model_final.pt \
    --episodes 5 --cmd "vx=0.4"

# 单次 + 渲染
uv run python scripts/fusion_control/mpc_sac/balance_mpc_sac.py --task walk_flat \
    --cmd "vx=0.4" --checkpoint <ckpt> --render video/fusion_control/mpc_sac/f1.mp4
```

## 路线

- ✅ F1b 平地: 稳定平衡 + 速度跟踪 + 高度(腿长)自适应
- ⏳ F2 粗糙地形: SAC 高层在 rough, 残差校正适应地形
- ⬜ F3 三栏对比: RL vs MPC vs 融合 (论文用)

## 参考

- 文献综述: `_devlog/xqrobotwl/common/2026-08-14/ref_mpc_rl_fusion_survey.md`
- AugMPC: arXiv:2603.10878 (IIT), 本地源码 `/home/robot/xlw/ibrido-containers/`
- 经典轨 (低层 MPC 移植源): `scripts/classic_control/mpc/`
