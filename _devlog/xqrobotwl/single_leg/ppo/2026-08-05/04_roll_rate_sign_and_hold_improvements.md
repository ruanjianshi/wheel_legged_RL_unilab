---
日期: 2026-08-05
来源: start_in_balance 训练 run8/run9 分析 (balance_complete 仍 0)
问题描述: start_in_balance 让机器人能对齐平衡倾角 (align_raw 0.97→0.38) 但自由轮
  离地只有 ~8%, 无法维持 0.5s 完整保持。奖励只奖瞬时对齐, 不罚震荡。
根因分析:
  1) 倒立摆保持需要**主动阻尼震荡**: 只奖对齐不够, 策略会漂离平衡角再被拉回,
     来回晃 (align_raw 0.97→0.37 掉), 自由轮时落时抬 (off ~8%)。
  2) ★ 我的 roll_rate 奖励**符号写反了**: `_reward_roll_rate` 返回 -rate (负误差),
     config scale 却是 -0.3 → (-rate)×(-0.3)=+0.3×rate, 变成**奖励抖动**!
     对齐惯例 (joint_action_rate 返回正误差 + scale 负值): 函数应返回 +rate,
     scale=-0.3 才是惩罚。
  3) 里程碑阈值 dot>0.93 太严 (倒立摆必然震荡, 偏差<22° 难持续), 放宽到 0.88。
解决方案:
  - `_reward_roll_rate` 改返回 +rate (正误差), config scale=-0.3 → 真惩罚
  - balance_upright scale 10→12, 加 roll_rate -0.3
  - 里程碑判据 dot>0.88 (偏差<28°)
修改文件:
  - src/unilab/envs/locomotion/xqrobotwl/single_leg.py (roll_rate 符号, 里程碑阈值)
  - conf/ppo/task/xqrobotwl_single_leg_flat/mujoco.yaml (balance_upright 12, roll_rate -0.3)
验证方法: 训练监控 roll_rate 应随迭代下降 (抖动减少), balance_complete 应首次触发
评估结果: run10 (修正符号后) 训练中
后续计划: 等 balance_complete 触发 → 确定性评估 → 启用过渡段课程
关联日志: 03 start_in_balance
---
