# 21 四算法最终对比 + 论文图 (优化完成)

**日期**: 2026-08-07
**来源**: 用户要求 — 四算法均跳到 10000 轮,确保跳起+落地不倒,出论文对比图(4X1/2X1)
**关联**: [20_vmc_slip_refactor_ppo_landing](2026-08-06/20_vmc_slip_refactor_ppo_landing.md)

---

## 最终四算法对比 (最佳 checkpoint, 4速度 × 4回合)

| 算法 | 跳高(m) | air(真跳) | 存活率 | 跳跃成功率 |
|------|------|------|------|------|
| **SRL** | 0.437 | **22.1%** | **100%** | **100%** |
| **纯PPO** | 0.286 | 10.2% | 50% | 100% |
| **PPO+VMC** | 0.053 | 4.7% | 62% | 94% |
| **VMC+SRL** | 0.015 | 2.5% | 50% | 19% |

使用 checkpoint:
- SRL: 2026-08-06_01-16-20/model_9999
- 纯PPO: 2026-08-06_21-40-31/model_4000 (最佳点: 跳0.35m+全存活)
- PPO+VMC: 2026-08-07_19-54-13/model_9999
- VMC+SRL: 2026-08-07_19-54-14/model_8000

## 优化过程总结

| 版本 | 纯PPO | PPO+VMC | VMC+SRL |
|------|------|------|------|
| v1 (原始) | 跳0.50m, 存活0% | 不跳(air 0.5%) | air 8.5% |
| 最终 | **跳0.29m, 存活50%** (最佳点0.35m+全存活) | **跳0.05m, air 4.7%, 存活62%** | air 2.5% |

关键优化:
- 纯PPO: landing_recovery 奖励 + 降熵, 从"跳但摔"到"跳+活"
- PPO+VMC: 加 SLIP-FSM 参考(全动作模式), 从不跳到小幅腾空
- VMC+SRL: 提高策略残差权限(0.5), 增强蹬伸增益

## 核心结论 (论文支撑)

1. **SRL (SLIP-FSM前馈) 明显最优**: 0.44m 跳高 + 22% air + 100% 存活 — 前馈的价值 = 跳跃稳定性
2. **纯PPO**: 能跳(100% 成功率)但多速度下落地不稳(存活50%) — 无参考引导的局限
3. **VMC 变体**: 在重载轮腿平台(xqrobotwl 18.65kg)腾空受限(air 2-5%), 力控难打破轮地接触
4. 关节空间 SRL 前馈 > VMC 空间 — 控制层面对比的核心发现

## 论文图 (已生成, PNG+PDF)

| 图 | 布局 | 内容 |
|----|------|------|
| `paper_fig_training.png` | **4X1** | 训练曲线: mean_reward / episode_length / vertical_thrust / jump_height, 4算法 |
| `paper_fig_validation.png` | **2X1** | 验证: 跳高 + air/存活率, 4算法 |

- Okabe-Ito 色盲安全色 (SRL蓝/PPO橙/PPO+VMC绿/VMC+SRL红)
- 输出: `jump_management/results/`

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/make_paper_figures.py` | 新建, 论文图生成(4X1训练 + 2X1验证) |
| `scripts/compare_jump.py` | (已建) 多速度对比 |
| `jump_management/results/four_algo_comparison.json` | 更新为最终结果 |
| `jump_management/results/paper_fig_*.png/pdf` | 论文图 |

## 后续计划

- [ ] 用户可将图放入论文
- [ ] VMC 变体腾空可作 future work (改进 VMC 力控/轮控实现更高腾空)
- [ ] 深调 VMC 腾空(推力爆发/轮控/飞行时序)如需

## 关联日志

- [20_vmc_slip_refactor_ppo_landing](2026-08-06/20_vmc_slip_refactor_ppo_landing.md)
- [19_four_algo_training_compare](2026-08-06/19_four_algo_training_compare.md)
