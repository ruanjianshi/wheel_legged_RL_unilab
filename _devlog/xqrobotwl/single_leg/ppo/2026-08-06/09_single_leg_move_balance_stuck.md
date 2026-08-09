---
日期: 2026-08-06
来源: single_leg_move RL 训练多次尝试后卡点 (独立 env 从零训练)
问题描述: 独立 env (30°侧压配重单轮) 从零训练, 策略学不会单轮平衡。
  训练指标: episode 36步(0.36s, tilt 倒) 或策略收回配重两轮(0.3s free_down 终止),
  wheel_off 退化到 0, balance_complete 从不触发。
根因分析 (逐项排查 + 实验):
  1) **物理稳定上限 ~1.35s**: 最优 PD (kp=60, sign=-1, 30°) 也只能保持 1.35s
     (腿屈-轮动耦合振荡, classic_control.py 确认)。单轮平衡需远超 PD 的控制。
  2) **RL 从零学不会倒立摆平衡**: 策略倾向于两轮(收回配重, 稳定但 0.3s 终止)
     或单轮 0.36s 倾覆。500/2000 iter 均无突破。
  3) 已修复的问题 (不是卡点根源):
     - force sensor 判接触不可靠 (自由轮离地仍 20-138N) → 改 L_hip_roll 判据
     - _physics_state 偏移 (state[0]=time, qpos 从 state[1] 开始)
     - 22° 侧压自由轮自然着地 → 30° 侧压 (96% 离地)
     - wheel_action_scale 10 太大 (0.05动作→0.5rad/s 噪声破坏平衡) → 3
  4) 尝试矩阵 (全部卡 ~1.5s):
     wheel_off 权重 4/15, balance_complete 50 (0.5/0.3s), free_down 终止 0.3s,
     init_noise_std 0.05/0.2, 侧压 22/25/28/30°, vx 0/±0.3, wheel_scale 10/3,
     500/2000 iter, L_hip_roll 动作映射
结论:
  - 30° 侧压配重单轮平衡, 物理稳定上限 ~1.35s (简单 PD)
  - RL 从零学不会 (倒立摆控制对 xqrobotwl 弱执行器太难)
  - **需要远超裸 PD / 裸 RL 的方法**: LQR/MPC 参考引导, 或结构/执行器改进
后续计划: 待用户决策 — LQR参考引导RL / 提高kp / 降低姿态难度 / 暂停
关联日志: 07 (经典控制物理验证), 08 (首训两轮作弊)
---
