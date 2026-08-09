---
日期: 2026-08-06
来源: 改进结构单轮平衡 RL 训练 v3 (位置drift惩罚) 训练至 model_3000
问题描述: 单轮平衡 RL 训练是否突破 — 验证 RL 能否学出 PD 做不到的耦合3D平衡控制.
根因分析/结果 (全部用修复后 eval: active mask 冻结已done env, 非累加假象):
  1) **model_1000**: eval 平均 **4.1s 平衡** (PD 0.86s 的 4.8倍). 失败模式=缓慢耦合
     tilt 漂移 (pitch/roll ~4s 衰减到 37° 阈值), 轮始终贴地, 补偿不足.
  2) **model_3000**: eval **20 env 全部 800/800 步满 8s 平衡** (100%), 左轮贴地 100%,
     pitch 对齐 0.995, roll 对齐 1.000 — 策略学会修正慢速倾斜漂移, 完全站稳.
  3) 渲染: video/single_leg/2026-08-06_RL单轮平衡_8s.mp4 (7.99s 满窗口)
  4) 训练指标 ~5.7-6.2s (含探索噪声).
  5) **最终 checkpoint 分析 (训练 4999 iter 完成)**: model_1000 4.1s / model_2000 3.9s /
     **model_3000 8s** / **model_4000 8s (20env 全跑满, pitch 0.995 roll 0.999)** /
     model_4999 回退 1.68s (PPO 后期不稳定). → **最佳 checkpoint = model_3000/4000**,
     部署用 model_4000. final_mean_reward 53.3 (best 72.7).
  6) **shell 脚本** (shell/xqrobotwl/, 遵循仓库约定):
     - launch_ppo_single_leg_unicycle.sh: 训练启动 (quick/full/resume/迭代数),
       与 launch_ppo_single_leg*.sh 家族一致
     - eval_ppo_single_leg_unicycle.sh: **可视化验证 = MuJoCo 原生窗口**
       (play_interactive.py, 与 eval_ppo_*.sh 家族一致: --keyboard/run_id/checkpoint),
       默认 model_4000 (最佳, 避免 model_4999 退化), --latest 用最新
     - 数值评估保留在 scripts/xqrobotwl/eval_single_leg_unicycle.py (独立工具)
     - 删除了多余的 play_ 脚本 (eval 就是可视化 play)
  7) **--keyboard 踩坑**: play_interactive 的 _policy_obs_contains_command 用 3D 探测命令
     [0.37,-0.23,0.19] 替换 vel_limit, 我的命令是 5D → reset 时 commands 变 3D 破坏 obs 帧
     (34≠36). 修: --keyboard 时加 +interactive.require_keyboard_command_obs=false
     (与 eval_ppo_backflip.sh 同). 已验证窗口正常打开.
结论 (诚实):
  - ✅ **改进结构 + RL = 完整 8s 单轮平衡实现**. 结构搜索 (CoM对齐+直腿+直配重+扭矩轮)
    给 RL 可控性基础, 位置drift惩罚 (非线速度) 是关键奖励设计 (v1 2.75s / v2 1.2s / v3 8s).
  - 从 v4 姿态 0.2s 塌落 → 结构搜索物理可行 → RL 8s 平衡, 整个闭环走通.
  - 这是本项目单轮平衡里程碑: PD 0.86s → RL 8s (9.3倍).
  - 剩余: 训练未完成 (model_3000 已达标), 待 5000 iter 完成 + 最终评估 + vx 移动 (第二阶段).
修改文件: scripts/xqrobotwl/eval_single_leg_unicycle.py (eval bug 修复: active mask)
验证方法: eval 修复后 20 env 全跑满 800 步 + 渲染视频
后续计划: 训练完成 → 最终 eval → 第二阶段 vx 前进/后退
关联日志: 21 (env实现), 20 (结构搜索), 19 (kp bug), 18
---
