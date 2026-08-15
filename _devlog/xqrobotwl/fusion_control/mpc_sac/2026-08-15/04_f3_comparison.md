# [04] F3 三栏对比交付 — RL vs MPC vs 融合 (论文「经典 vs 学习 vs 融合」)

**日期**: 2026-08-15
**状态**: 完成 — 三栏对比表 (平地 + 粗糙地形)
**关联**: [[02_f1b_sac_training_flat]], [[03_f2_rough_fusion]], [[INDEX.md]]

---

## 平地 (walk_flat) — 融合最优

| 方法 | 存活率 | vx_rmse | 说明 |
|---|---|---|---|
| RL 端到端 (walk_flat 重训) | 100% (30s 站立) | **0.503** | 能站不走, 速度追踪失效 |
| 经典 MPC (P2) | 100% | **0.087** | 经典基线, 速度跟踪 |
| **MPC×SAC 融合** (F1b) | 100% | **0.039** | 残差学习补偿滞后, 跟踪最优 |

**平地结论**: 融合 < MPC < RL。融合 (0.039) 优于纯 MPC (0.087) — 残差 RL 学会补偿 MPC
滞后 (超发令), 速度跟踪改善 55%。RL 端到端重训模型退化为"站立不动" (vx_rmse 0.503)。

## 粗糙地形 (gentle rough) — RL 最优, 融合未超 MPC

| 方法 | 存活 (avg) | 说明 |
|---|---|---|
| RL 端到端 (walk_rough 重训, 硬地形训练) | **10.4s** | 硬地形训练 → 缓坡泛化最佳 |
| 经典 MPC (P4, 缓坡) | 3.3s | 地形是 MPC 短板 |
| **MPC×SAC 融合** (F2b) | 2.4s | 略低于 MPC, 未超过 |

**粗糙结论**: RL > MPC > 融合。命令级残差融合无法补偿地形扰动; MPC 低层地形适应是瓶颈。
融合的价值在平地 (指令跟踪), 不在地形。

## 综合结论 (论文用)
1. **MPC×SAC 残差融合的价值**: 平地速度跟踪 (vx_rmse 0.087→0.039, 改善 55%) —
   RL 补偿 MPC 模型误差/滞后的经典例证。
2. **融合的局限**: 命令级残差不解决地形扰动 (粗糙地形 2.4s < MPC 3.3s < RL 10.4s)。
   地形自适应需要腿级/接触级融合 (AugMPC 接触显式) 或地形感知 obs。
3. **RL 端到端的现状**: walk_flat 重训退化为站立 (0.503); walk_rough 硬地形训练泛化好 (10.4s)。

## 数据来源
- RL flat: `tools/xqrobotwl/eval_walking.py` (30s, vx_rmse 0.503)
- MPC flat: 经典轨 report (P2 vx_rmse 0.087)
- 融合 flat: F1b report (vx_rmse 0.039, 5ep 100%)
- RL rough: gentle rough 6ep 10.4s
- MPC rough: gentle rough 10ep 3.3s
- 融合 rough: gentle rough 10ep 2.4s (F2b)

## 后续
- 论文「经典 vs 学习 vs 融合」表格素材已齐
- 融合改进方向: 腿级残差 / 地形感知 obs / 接触时序 (见 F2 devlog)
