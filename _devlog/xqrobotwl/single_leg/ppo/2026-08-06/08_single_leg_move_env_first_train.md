---
日期: 2026-08-06
来源: 用户重构要求(可移动单轮平衡) + plan 阶段二 — 独立 env 从零训练
问题描述: 依据经典控制物理验证(devlog 07), 构建独立可移动单轮平衡 env。
  用户要求: 单腿前进/后退/转向, 独立开发不牵扯后空翻/walk-flat, 从零训练。
根因分析/设计:
  - 经典控制发现(07): kp=60 柔性最佳, 22° 侧压折中, 最优策略=微扰动保持
  - 独轮车式: 支撑轮 R_wheel 控 pitch+移动, 自由腿配重 L_hip_roll 控 roll
  - 与 single_leg.py 完全独立 (继承 XqRobotWLWalkFlatEnv 最底层, 无 FSM/后空翻遗产)
实现:
  - 新建 src/unilab/envs/locomotion/xqrobotwl/single_leg_move.py
    - 22° 侧压平衡位 reset (up_ref=[0,sin22°,cos22°])
    - step 钉住: 自由腿 pitch/knee + 支撑腿 pitch/knee; 放开配重 roll + 双轮
    - 奖励: tracking_lin_vel + balance_upright + wheel_off + fold_pose + roll_rate + alive
    - 宽容终止 (侧压平衡位本身 tilt≈22°)
  - 新建 conf/ppo/task/xqrobotwl_single_leg_move/mujoco.yaml (从零, init_noise_std 0.05)
  - 新建 launch_ppo_single_leg_move.sh / eval_single_leg_move.py
首次训练 (quick 500 iter, 1024 env, mps):
  - ✅ 训练管线跑通, reward 1.4→69.1, episode 长度 80→800 (全部存活)
  - ❌ **自由轮离地率 0%** — 策略学会放下自由腿两轮支撑作弊 (L_hip_roll 是 RL 自由配重通道)
  - ❌ vx 未跟踪 (vx_cmd=0.3 实际 0.002) — 策略专注两轮保持
  - reset 运动学自由轮离地 +0.156m ✓, 但策略训练中学着放下
修正 (防作弊):
  - wheel_off 权重 4→15 (两轮支撑惩罚)
  - 新增 balance_complete 奖励: 连续单轮+平衡+高度 0.5s 一次性大奖 50
    (env 维护 _single_leg_hold 计数, 达标给奖)
  - 重训 quick 验证 wheel_off 是否提升
修改文件:
  - src/unilab/envs/locomotion/xqrobotwl/single_leg_move.py (奖励/单轮计数)
  - conf/ppo/task/xqrobotwl_single_leg_move/mujoco.yaml (wheel_off 15, balance_complete 50)
  - scripts/xqrobotwl/eval_single_leg_move.py (确定性评估)
  - shell/xqrobotwl/launch_ppo_single_leg_move.sh
  - src/unilab/envs/locomotion/xqrobotwl/__init__.py (注册)
验证方法: smoke test (reset 22° 平衡位 + 零动作 step) + quick 训练 + 确定性评估
评估结果: 首次训练两轮作弊; 修正后重训中
后续计划: 重训后评估 wheel_off/离地率 → full 训练 → vx 移动 → 渲染视频
关联日志: 07 (经典控制物理验证), 01-06
---
