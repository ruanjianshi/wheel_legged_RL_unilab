# 05 对称几何下全部重训 13 个 RL 任务 (启动)

## 日期
2026-08-14

## 来源
几何对称化 (devlog 03/04) 改变了 `xqrobotwl.xml` 物理, 13 个任务训练模型均在**不对称几何**下训练。
用户要求: **先全部重新跑训练** (对称几何下重训)。

## 修改了什么
无代码修改 — **启动 13 个任务全量重训** (新几何, 全新 run, 不改 config)。

| # | 任务 | 算法 | 入口 |
|---|---|---|---|
| 1-11 | walk_flat/walk_rough/toe_walk/jump/jump_srl/jump_vmc/jump_srl_vmc/backflip/single_leg×3 | PPO | `train_rsl_rl.py task=<task>/mujoco` |
| 12 | fall_recovery | CPO | `train_cpo.py` |
| 13 | stairs | NP3O | `train_np3o.py task=xqrobotwl_stairs/mujoco` |

启动方式: `nohup ... > logs/train/<task>.log 2>&1 &`, CUDA_VISIBLE_DEVICES=0 (GPU0 only)。

## 资源情况 (如实)
- 机器 CPU 被其他任务占满 (AugMPC 集群/comfyui/obs, 负载曾 167/32核)
- 13 任务仍以 1024 envs × 10000 iters 分批启动, 在争抢下迭代 1.0-2.6s/轮 (健康)
- 后续负载降至 60, 迭代加快 (walk_flat 2.2s→0.6s→1.4s)

## 监控
- 🔴 错误监控: 13 个训练日志的 Traceback/OOM/Killed 实时告警
- 📦 checkpoint 监控: 每 1000 轮新 model_N.pt 自动通知 → 届时按 CLAUDE.md §1.2 跑评估
- 所有任务起始 checkpoint (model_0.pt) 已保存, 训练无错误

## 参数调整
无 (沿用各任务 config 默认: 1024 envs, max_iterations 10000, save_interval 1000)

## 根因分析
几何对称化 (左右轮镜像) 改变倒立摆支点/轮距 → 旧模型 (不对称几何训练) 在新几何上表现可能退化 → 需重训。

## 验证方法
- 冒烟: walk_flat num_envs=64 max_iter=3 训练通过, 生成 run 目录 + 渲染视频 (已清理)
- 全量启动后: 13 任务全部有训练轮数增长, 无错误
- 每 1000 轮评估 (checkpoint 监控触发)

## 后续计划
- 等待各任务 checkpoint, 每 1000 轮评估 (对比对称几何下指标)
- 训练完成后对比旧模型 (不对称几何) 指标
- P4 rough 地形稳定性 (经典控制遗留)

## 关联日志
- [[03_wheel_symmetry_fix]] 几何对称化
- [[04_verify_all_13_tasks_model]] 13 任务模型核对
