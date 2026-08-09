# [12] 用户反馈: 跳到高度但站不稳 → settle 平衡奖励 + 1024 envs 从零训练

**日期**: 2026-08-07
**来源**: 用户看交付视频后反馈 "有跳动达到预期高度, 但根本站不起来, 平衡稳定"; 要求并行环境 1024 个 + 从零训练
**关联**: [11_delivery_single_robot_recovery](11_delivery_single_robot_recovery.md)

---

## 问题描述

交付视频显示机器人能到 0.5-0.67m (预期站立高度), 但无法稳定站住。
用户反馈: 有跳动达到高度, 站不起来, 平衡不稳定。

## 根因分析

**复现数据** (model_15000, 10 episodes 轨迹):
- 轮离地仅 0-5 步 (不是跳) — 是**推上去后倾倒/塌缩**
- 站立条件 (z>0.45 + 直立>0.85 + 双轮) 仅维持 6-50 帧 (ep5 临界 50 帧=0.5s) 就丢
- ep5: max_z=0.67, 站住帧=50; ep9: max_z=0.62, 站住帧=6

**结论: 策略会"推上去"但不会"站稳"** — 缺失平衡技能:
1. **rise_vel 奖励在顶部也生效** — 到站立高度后继续奖推力 → 冲过头/顶部跳动
2. **没有任何"稳定站立"奖励** — 直立+高度到了, 但静止平衡无激励; walk 热启动的平衡技能被恢复训练覆盖
3. base_height 对超冲 (0.67 vs 0.55) 无惩罚 (clip 到 1.0)

## 解决方案

1. **新增 `settle` 稳定站立奖励** (scale 15): 直立 × 接近站立高度 (|z-0.55|<0.15) × 静止 (垂直速度低)
   — 直接教"站稳", 补缺失的平衡技能; 躺地/跳动/倾倒时≈0
2. **rise_vel 门控**到 base_z < 0.45 (`rise_vel_height_cap`) — 起身过渡期奖推, 顶部归零 (防跳动)
3. **1024 并行环境** (用户要求): 验证 MPS 可行且快 (296k env步/秒 vs 128 时 5k)
4. **从零训练** (用户要求): `load_run=-1` 随机初始化, 不继承半撑/跳动行为

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | 新增 `_reward_settle` (直立×高度×静止); `rise_vel_height_cap` 配置; rise_vel 门控 base_z<cap; `info["abs_rise_vel"]` |
| `conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | scales_ru/rs 加 `settle: 15.0`; `rise_vel_height_cap: 0.45` |
| `scripts/xqrobotwl/eval_fall_recovery.py` | 新增"平均最长连续站立时间"指标 (直接量化站住稳定性) |

## 验证方法

- rise_vel 门控: base_z=0.59 时 rise_vel=0 (门控生效), 低处正常 ✓
- settle 计算: 注册无崩溃 ✓
- 1024 envs MPS: 创建 0.1s, step 3.5ms, 296k env步/秒 ✓

## 后续计划

- [ ] 从零训练 (run `2026-08-07_16-27-38_mujoco`, 8000 iter, 1024 envs) ~2.4h
- [ ] 每 1000 iter 无辅助评估 (新增站立保持时间指标)
- [ ] 稳定站立 (保持>0.5s 且恢复率>50%) → 渲染视频 → 停止

## 关联日志

- [11_delivery_single_robot_recovery](11_delivery_single_robot_recovery.md) — 前次交付 (站不住)
- [10_phase2_forcefree_risevel](10_phase2_forcefree_risevel.md) — rise_vel 突破 (推力会了, 平衡缺)
