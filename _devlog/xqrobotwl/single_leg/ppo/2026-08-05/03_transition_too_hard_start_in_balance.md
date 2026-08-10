---
日期: 2026-08-05
来源: 单腿平衡训练 run5/run6 不收敛 (balance_complete 恒 0)
问题描述: 即使修复 roll_ref 符号/奖励梯度/折腿缓和, 训练到 1000 iter 仍无一次 balance_complete。
根因分析:
  1) **折腿过渡本身太难**: 折腿时自由腿缩短, 支撑腿膝 kp=60 撑不住 18.65kg 单腿负载,
     base 塌缩到 z≈0.30 (< balance_complete 的 height_ok 阈值 0.40), 且 pitch 俯冲到 -58°。
     raw MuJoCo 实测: 折叠+支撑腿伸直+强轮PD 仍 FALL at ~1.2s。
  2) **balance_complete 阈值(高度>0.40)物理上在过渡中不可达**: 过渡塌缩 z<0.40 → 永不触发。
  3) **静态平衡位其实可达**: fold_knee=0.60 时 lean=-32° 存在平衡 (|CoM-Rwheel|=0.006), 
     但动态过渡到那里做不到 — 过渡需要同时折腿+支撑腿伸直+侧倾, 协调太难。
解决方案:
  - ★ start_in_balance=True: reset 直接置 P1 折叠平衡位 (base 侧倾 -32°, 自由腿收膝 0.60,
    支撑腿伸直), FSM 直接状态1。RL 先学"保持", 过渡段以后作为扰动再学。
    与后空翻 warmstart 同理: 先让策略在一个已知好姿态上站稳, 再学动作本身。
  - 过渡段(站立→折腿)保留 start_in_balance=False 路径, 等保持学会了再启用 (课程)。
修改文件:
  - src/unilab/envs/locomotion/xqrobotwl/single_leg.py (DR provider 加 start_in_balance reset 姿态,
    _reset_done_envs FSM=1)
  - conf/ppo/task/xqrobotwl_single_leg_flat/mujoco.yaml (start_in_balance: true)
验证方法: 训练监控 balance_complete 应 >0; 确定性评估 one-wheel 比例应上升
评估结果: run6 (过渡模式) 1000 iter balance_complete=0, one-wheel 21.5%; start_in_balance 起步中
后续计划: 先学保持 → 确定性评估 → 再启用过渡段课程
关联日志: 02 env 设计, 04 训练结果
---
