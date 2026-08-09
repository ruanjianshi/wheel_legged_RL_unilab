# [05] 首次训练评估 — 管线通但恢复未成形 (局部最优 0.26m)

**日期**: 2026-08-07
**来源**: 首次完整训练 (warmstart 自 walk + CPO, 300 iter) 后的确定性评估
**关联**: [01_fsr_design](01_fsr_design.md) / [04_cpo_conf_dir_and_value_scale](04_cpo_conf_dir_and_value_scale.md)

---

## 问题描述

首次训练: warmstart 自 walk, CPO, 128 envs, 300 iter (~4 min on M4)。
训练曲线: reward 0.22→22.4, ep_len 507→754。但确定性评估发现**机器人没有真正站起**:
- 恢复率 (base_z 达 0.55m) = **0%**
- 平均最大高度 = **0.26m** (躺地~0.15, 站立~0.55)
- 平均 ep_len 601 步

## 根因分析

**局部最优**: 机器人撑起到 0.26m ("蟹式半撑"姿态):
- 0.26m > idle 终止阈值 0.16m → 躲过贴地终止
- base_height 奖励 exp(-8.3×(0.26-0.35)²)≈0.94, 与 0.35 目标几乎同分 → 梯度太平
- 于是停在 0.26m 混 alive 奖励到 episode 截断 (reward 22 里大部分是 alive)

次要因素: 300 iter×128 env 远小于论文 (8000×4000); Fmax=60N 在 0.26m 处
只剩 22N 辅助, 跨不过 0.26→0.35 的坎。

## 解决方案 (候选, 未实施)

1. **提高 idle 终止阈值** 到 ~0.25 (base_z < 0.25 持续 6s 终止) — 杀掉"撑到 0.26 混分"
2. **加强 height 奖励梯度**: 对 h_cmd1 以下的偏离用线性惩罚 (如 max(0, h_cmd1-h))
   替代 exp (exp 太饱和)
3. **提高 Fmax** (60→100-120N): 帮助跨过 0.26→0.35 坎 (训练期辅助, 后撤除)
4. **延长训练**: 2000+ iter (论文 8000), 或加大 env 数

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `scripts/xqrobotwl/eval_fall_recovery.py` | 新建: 确定性恢复评估 (恢复率/保持率/最大高度) |

## 验证方法

`uv run mjpython scripts/xqrobotwl/eval_fall_recovery.py --run <run_dir>`

## 评估结果

- 训练曲线: reward 0.22→22.4 ✓, ep_len 507→754 ✓ (看起来在学)
- **实际恢复: ❌ 0%** (max height 0.26m) — 局部最优, 非学习失败
- CPO 约束: constraint_value ~410 稳定, viol ~0.004 (管线正常)
- **教训**: 训练 reward/ep_len 会因局部最优虚高, 必须确定性评估看真实恢复率

## 后续计划

- [ ] 实施候选修复 (idle 阈值/height 梯度/Fmax) — 单一变量对照
- [ ] GPU 服务器长训 (2000+ iter)
- [ ] 恢复成形后: 力撤除 (force_end_iters) 后成功率验证

## 关联日志

- [04_cpo_conf_dir_and_value_scale](04_cpo_conf_dir_and_value_scale.md)
