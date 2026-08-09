---
日期: 2026-08-05
来源: 单腿平衡 env 开发 (任务 12-14)
问题描述: 首个训练 run 平衡奖励全 0 — 机器人不会侧倾压质心到支撑轮。
根因分析:
  1) **roll_ref 符号反了**: P1 平衡位 qpos lean=-28° → up_y=+0.469 (机身向支撑轮侧倾),
     但 env 最初用 `_ROLL_REF_RAD=-28°` 奖励 up_y=-0.47 — 往支撑轮**外侧**倒, 方向反了。
  2) **奖励梯度太平**: dot-product 奖励直立给 8.83/10, 完全侧倾给 10/10, 梯度太弱,
     策略没有动力真的侧倾 (侧倾是让 CoM 落支撑轮的关键)。
  3) **折腿过渡不稳**: 自由腿折膝缩短, 若支撑腿不伸直 + 机身不侧倾, base 会 squat 塌缩
     (base_z 0.65→0.27), 自由轮无法离地 → wheel_off 恒 0。
解决方案:
  - `_ROLL_REF_RAD = +28°`, `_reward_balance_upright` 改用 `up_y/sin(28°)` 线性梯度
    (直立=0, 完全侧倾=1, 比 dot 平缓曲线陡得多)。
  - FF 只脚本化自由腿折腿 (0-2); 支撑腿(3-5)+轮子(6-7) 全部状态策略自由 —
    侧倾 + 支撑腿伸直 + 轮式俯仰都必须闭环反馈, 不能脚本化。
  - 折腿过渡 0.6s → 1.0s (给 RL 时间把重量移到支撑腿)。
修改文件:
  - src/unilab/envs/locomotion/xqrobotwl/single_leg.py (roll_ref 符号, 奖励形状, FF 范围, FSM 时长)
验证方法: 训练监控 reward/balance_upright 应 >0 且随迭代上升
评估结果: 修正前 balance_upright=0; 修正后 run3 迭代 196 已达 2.9 (scale 10), 但 wheel_off 仍 ~0
后续计划: 训练跑通 balance_complete>0; 再确定性评估
关联日志: 01 P1 可行性, 03 训练结果
---
