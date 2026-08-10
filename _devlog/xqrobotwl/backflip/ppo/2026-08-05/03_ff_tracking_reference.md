# 03 — ff_tracking 参考跟踪 (文献 OPT-Mimic, 防甩腿/麻花)

## 日期

2026-08-05

## 来源

用户反馈: 后空翻腿扭成麻花, 落不了地。文献分析: 结果奖励(flip_progress/flip_complete)不约束动作质量, 策略靠甩腿伪造旋转 — 正是 [CAV论文](https://deep-paper.org/en/paper/2505.12222/) 的 BAV 奖励缺陷。

## 根因

- 开环 ff 是干净翻转参考轨迹
- 闭环只用结果奖励, 策略无约束去复现干净翻转 → 甩腿/过度旋转
- [OPT-Mimic](https://www.cs.ubc.ca/~van/papers/2022-opt-mimic/index.html): 开环轨迹 + 模仿跟踪是标准解法

## 解决方案

1. **`ff_tracking` 奖励**: 翻转中(蹬/飞/展)惩罚策略输出偏离 ff (防甩腿/麻花)
2. **flip_progress 封顶**: 超过 360° 不再奖励 (防过度旋转)
3. **单次翻转**: flip_done 抑制重复触发 (之前已加)

## 修改文件

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotwl/backflip.py` | 加 _reward_ff_tracking, flip_progress 封顶, step 存 policy_actions |
| `conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml` | ff_tracking=1.0 |

## 验证方法

续训后: 翻转中策略动作小 (ff 主导) → 腿不甩/不麻花, 保持开环干净翻转。

## 后续计划

- resume 续训, 渲染验证腿部姿态
- 达到开环质量后进入 P4

## 关联日志

- [02_flip_progress_measurement_fix](2026-08-05/02_flip_progress_measurement_fix.md) — 测量修复
