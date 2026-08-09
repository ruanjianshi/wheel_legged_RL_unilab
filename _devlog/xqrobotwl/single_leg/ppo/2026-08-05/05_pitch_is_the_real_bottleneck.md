---
日期: 2026-08-06
来源: run11 (start_in_balance, iter 2356) 平台化分析
问题描述: 训练 2000+ iter 后 align~0.60 / off~0.09 / comp=0, 不再改善。机器人学会部分
  对齐+偶尔抬轮, 但学不会稳定保持。
根因分析:
  1) **pitch(俯仰倒立摆) 是单腿平衡的真正难点**: 详细 trace 显示崩溃是 pitch 从 -5° 突冲到
     +48° (t=0.4→0.62s), 不是 roll。roll 有静态平衡位支撑(CoM 距轮 0.006m), pitch 是纯
     动态倒立摆, 必须靠轮子主动稳住。
  2) **warmstart 的 pitch 能力不迁移**: walk 模型学了双轮 pitch 控制, 但单腿时动力学不同
     (单轮+折腿+侧倾28°), 继承的 pitch 权重失效。
  3) **R_hip_roll 不是 roll 主执行器**: 平衡位 R_hip_roll≈-0.10, 大幅摆动反而破坏
     |CoM-Rwheel| (0.025→0.307)。roll 主要由侧倾姿态维持, hip_roll 只小幅微调。
  4) balance_upright 奖励的 pitch 分量权重不足 (up_ref 里 pitch 用 cos28°≈0.88)。
解决方案: 未定 — 需重新设计 pitch 控制信号 (可能: 强化 balance_upright 的 pitch 分量 /
  单独 pitch 奖励 / 提高轮子控制权)
修改文件: 无
验证方法: 详细 trace 定位失败模式 (pitch 发散, 非 roll)
评估结果: run11 平台化: align 0.60/off 0.09/comp 0
后续计划: 需决策 — 强化 pitch 信号 or 换策略
关联日志: 01-04
---
