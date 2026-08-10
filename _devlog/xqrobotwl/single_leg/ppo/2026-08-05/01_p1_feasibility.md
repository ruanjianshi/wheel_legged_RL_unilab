---
日期: 2026-08-05
来源: 用户发起单腿平衡(单轮支撑)高难度动作开发
问题描述: 后空翻已交付, 用户想开发单腿平衡。需先回答物理可行性再进 RL。
根因分析: 单腿平衡与后空翻本质不同 — 后空翻是弹道(开环可脚本化), 单腿平衡的**保持**是倒立摆调节(必须闭环反馈)。P1 只能静态验证三点。
解决方案: 新增 scripts/xqrobotwl/single_leg_balance_feasibility.py 验证三件事。
修改文件:
  - scripts/xqrobotwl/single_leg_balance_feasibility.py (新增)
验证方法: uv run python scripts/xqrobotwl/single_leg_balance_feasibility.py
评估结果:
  1) 静态平衡位可达 ✅ — 收膝折腿(L_knee=0.87, L_hip_pitch=0.30) + 机身侧倾 -28° +
     支撑腿微调 → CoM 距支撑轮仅 0.026m, 轮着地(z=0.104), 自由轮离地 0.377m。
  2) FF 折腿过渡稳定 ✅ — 站立收膝折腿 0.6s, CoM 横向最大偏移 0.102m < 两轮半宽 0.19m,
     过渡过程不甩出支撑多边形, 不会提前倒。
  3) 横滚控制权充足 ✅ — 支撑腿 R_hip_roll -0.10→-0.90 使支撑轮横向扫 0.31m 且全程着地,
     可覆盖横滚质心扰动。
  结论: 单腿平衡物理可行。折腿机制必须用**收膝**(knee bend), 不用**髋外展**(hip_roll):
     收膝抬轮且 CoM 横向不动; 髋外展把 CoM 甩离支撑轮(dy -0.19→-0.36), 过渡即倒。
后续计划: 进入 RL — FSM(站立→折腿FF→单轮平衡RL→落腿FF→站立), RL 学横滚反馈。
关联日志: 后续 env/config/训练 devlog
---
