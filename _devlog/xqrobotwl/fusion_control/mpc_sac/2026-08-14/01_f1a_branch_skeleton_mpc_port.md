# [01] F1a 融合控制分支骨架 + MPC 移植 (MPC×SAC)

**日期**: 2026-08-14
**状态**: 完成 — 分支骨架 + 移植 MPC 数值回归 + 冒烟全链路通过
**关联**: [[../../common/2026-08-14/ref_mpc_rl_fusion_survey.md]] (AugMPC 调研), [[../INDEX.md]]
**关联经典轨**: [[../../classic_mpc/INDEX.md]] (MPC 移植源)

---

## 来源
用户需求: 参考 `/home/robot/xlw` 的 **IBRIDO/AugMPC** (高层 RL 决策 + 低层 MPC 执行),
移植到本项目, 在 `scripts/fusion_control/` 构建融合控制分支, **按"MPC 与哪个 RL 算法融合"命名** = `mpc_sac`,
**独立任务自包含** (自带 MPC, 不依赖经典轨 MpcController)。

## 架构 (AugMPC → 两轮足)
```
高层 SAC 策略 ── 命令 [vx,vyaw,height] ──→ 低层分支 MPC ── a8 ──→ env.step
  (obs: θ,θ̇,v,ω_z,base_z,期望指令,上一动作)
```
MPC 扛平衡 (两轮足最难的欠驱动), 高层 SAC 只学"发什么命令" → 收敛快。

## 分支结构 (自包含独立轨)
```
scripts/fusion_control/mpc_sac/
├── mpc/          ★ 分支自带线性 MPC (移植, 不 import 经典轨)
│   ├── qp.py        Hildreth QP (build_mpc_matrices/solve_qp/dlqr_riccati)
│   ├── dynamics.py  轮速命令模型 + 黑箱模型加载
│   └── controller.py 低层 MpcController (sagittal QP + act 骨架)
├── sac/          ★ 紧凑 SAC 高层策略 (自包含)
│   ├── networks.py  GaussianActor + TwinQ
│   ├── replay.py    回放缓冲
│   └── trainer.py   SAC 更新 + obs 归一化 + 温度自学习
├── config.py     MpcSacConfig (conf/fusion_control/mpc_sac 自包含)
├── env.py        build_env/read_sensors 向量化 (conf_dir 自包含)
├── obs.py        高层 obs / denorm_action / 期望采样 / reward
├── policy_loader.py  HighLevelPolicy 推理包装
├── controller.py MpcSacController (SAC→cmd→MPC→a8)
├── meta_env.py   MpcSacMetaEnv (向量化训练 env, auto-reset)
├── runner.py     run_episode (record 对齐 common.metrics)
├── metrics.py    融合指标 (θ/v 违犯率, cmd_track_err, solve_ms)
├── train_mpc_sac.py  SAC 训练入口
├── eval_mpc_sac.py   批量评估 CLI
└── balance_mpc_sac.py 单次运行+渲染 CLI
conf/fusion_control/mpc_sac/   robot/commands/config/task 自包含
_devlog/xqrobotwl/fusion_control/mpc_sac/ + INDEX
shell/xqrobotwl/fusion_control/mpc_sac/
```

## 移植 MPC (数值回归)
- qp.py/dynamics.py/controller.py 逐行移植自经典轨 (sagittal QP + act 骨架), 独立拥有。
- **回归门禁**: 移植版 vs 经典轨 MpcController — A_d/B_d/G 最大差 **0.000e+00**, `_sagittal_u` 输出 **diff=0.000e+00** ✅
- 黑箱模型 `logs/classic/mpc_plant_bb.npz` 共享只读数据 (§3.2)。

## 冒烟 (数据)
| 检查 | 结果 |
|---|---|
| config 加载 (obs 11D / 3D action) | ✅ |
| meta_env (2 env) reset/step, 随机命令 8 步 | ✅ 存活, base_z≈0.512-0.517 |
| 短训练 60 iter (4 env, cuda) | ✅ 训练循环 + 存档 |
| 推理链 balance_mpc_sac (8s, 未训练模型) | ✅ 存活 8s, gyro_rms 0.22, θ 违犯率 0, solve 0.11ms |
| 移植 MPC 回归 | ✅ 数值一致 |

未训练模型 vx_rmse 0.398 (高层随机发令) — 预期, F1b 训练解决。

## 修复记录
- ROOT 深度: 融合分支在 `scripts/fusion_control/mpc_sac/` 深 4 层 (经典轨 3 层) → dynamics parents[3]→[4], env CONF_DIR.parents[3]→[2]。
- SAC build_mlp 用 F.relu (函数) 非 nn 模块 → 改 nn.ReLU/ELU/Tanh。
- fk_legs 单 env → 分支向量化 `_fk_legs_batch`。

## 后续计划
- F1b: 平地 SAC 高层训练 (num_envs=64, max_iter=3000) → 验证存活≥95%/vx_rmse<0.1
- F2: rough 融合; F3: 三栏对比表

## 关联
- 参考文档: [[../../common/2026-08-14/ref_mpc_rl_fusion_survey.md]]
- 经典轨: [[../../classic_mpc/INDEX.md]]
