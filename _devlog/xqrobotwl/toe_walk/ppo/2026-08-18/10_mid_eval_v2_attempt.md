# 10 — 双模式训练 v1 中间评估: 站立达标但抬腿未学到 → v2 配方 (推力翻倍+课程压缩)

## 日期
2026-08-18

## 来源
v1 训练 (2026-08-18_02-13-16 run, 10k iter, 课程 [2000,5000]) 中途评估 (CLAUDE.md §1.2 每千轮评估) 发现问题。

## 做了什么

1. 新增确定性双模式评估工具 `tools/xqrobotwl/verify_toe_walk_mode.py` (站立3s→抬腿前进→侧移→转向→后退→切回站立 序列 + 站立/交替/对称/追踪/切换判定) 与渲染工具 `tools/xqrobotwl/render_toe_walk_mode.py`
2. 对 v1 run 的 model_5000 中间评估
3. 依据证据终止 v1 训练, 调整 conf 重启 v2 训练

## 中间评估结果 (model_5000, 8-18)

| 项 | 结果 | 判定 |
|---|---|---|
| 站立模式 | linvel_xy=**0.048** m/s (<0.2), gyro=**0.135** (<1), h=**0.516**m, 漂移 0.03m | ✅ **PASS (重大进展: 双模式站立微动达标)** |
| 抬腿模式 | 切换后 L/R 轮离地 = **0/400, 0/400**; 纯 mode=1 从头跑 L/R=0/500 | ❌ **完全未抬腿** |
| 指令追踪 | forward rmse 0.243 (偏大), 其余段 0.02-0.19 | ⚠️ 中 |
| 切换稳定 | 0.5s 内不倒 | ✅ PASS |

训练日志佐证: Stage1 (0-2000, 纯站立) swing_lift=0 (门控正确), ep_len ~960-990 ✅;
Stage2 (2000-5000, 纯抬腿) swing_lift 仅 0.06~0.11 (scale=30 → ~2-3/步, 历史 v1 为 0.48);
Stage3 (5000+, 混合) **掉到 0.024** — 混合后抬腿行为进一步稀释。

## 根因分析

1. **站姿先验锁死**: Stage1 纯站立 2000 iter 让网络充分收敛到"站立" (1200 iter 时 ep_len 已 958 — 站立学得快, 不需要 2000 iter)。Stage2 突然全部切抬腿, 策略需要破坏既有参数, 3000 iter 不够。
2. **推力不足**: phase_swing_lift 30 相对 base_height(-100)/orientation(-30) 的抬腿重心侧移惩罚太弱, 策略选择 "mode=1 也不抬腿" 的白拿路径 (alive+tracking+sym=0 罚)。
3. **课程阶段切换太晚**: Stage2 结束时抬腿未成型, Stage3 混合中 50% 站立样本进一步稀释。

## 参数调整好坏 (v2)

| 参数 | v1 | v2 | 理由 |
|------|----|----|------|
| phase_swing_lift | 30 | **60** | 摆地离地推力翻倍 |
| phase_knee_lift | 15 | **20** | 弯膝协同加强 |
| swing_contact_penalty | 5 | **10** | 摆地着地惩罚翻倍 (防白拿) |
| mode_curriculum_iters | [2000,5000] | **[1800,5000]** | 站立学得快, 压缩 200 iter 给抬腿 |
| 其余 (stand_still -5, lift_symmetry -20, action_scale 0.18, tracking 4.0) | — | 不变 | 站立已验证达标 |

## 验证方法
- 每 1k iter 用 `verify_toe_walk_mode.py` 评估 (重点: 抬腿段 L/R 离地次数、swing_lift 训练曲线)
- Stage2 结束 (5000) 时抬腿段离地率必须 >0 且 swing_lift 训练值 >0.25 才继续
- 若 Stage2 仍不抬: 进一步 Phase B (参考轨迹强制交替) 或 把 Stage2 拉长到 [1500,7000]

## 后续计划
- v2 训练 (GPU1, 10k iter) 收尾后: 完整 verify + 渲染视频 + 达标备份
- devlog 更新于 v2 关键里程碑 (3000/5000/9999)

## 关联日志
- [09_mode_switch_env_interactive](2026-08-18/09_mode_switch_env_interactive.md) — v1 环境/交互实现
- [08_verify_alternation_symmetry](2026-08-18/08_verify_alternation_symmetry.md) — 抬腿退化根因 (对称约束来源)