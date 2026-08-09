---
日期: 2026-08-06
来源: 用户反馈单腿平衡姿态设计 — 机身侧压 + 自由腿调节平衡
问题描述: 原 P1 方案(收膝折腿+28°侧倾, 自由腿钉死)训练 2000+ iter 平台化, balance_complete=0。
用户明确姿态意图: 支撑腿竖直轮着地, 机身通过髋roll角旋转压向支撑腿, 自由腿用于调节平衡。
根因分析:
  1) 详细 trace 显示原方案崩溃是 pitch 发散(-5°→+48°), 非 roll。warmstart pitch 不迁移单腿。
  2) 物理验证: 机身侧倾 ~30° 是 CoM 落支撑轮的甜点位 (|dy|=0.000), 超过 40° CoM 冲过轮。
  3) 机身"立起 90°"不可行 — 高重心倒立摆, 动态无法保持 (数值爆炸)。
  4) 用户意图实质 = 30° 侧压 + 自由腿 L_hip_roll 展开当配重 (像走钢丝者持杆) — 自由腿
     是唯一能主动移 CoM 的通道, 原方案钉死它导致 RL 无法用配重调节。
解决方案 (用户选择 "改env: 30°侧压+自由腿配重给RL"):
  - _ROLL_REF_RAD 28°→30° (CoM 精确落支撑轮)
  - _FOLD_KNEE 0.60→0.30, _FOLD_PITCH 0.20→0.10 (自由腿微屈不深折)
  - _FREE_LEG_ROLL_INIT=-0.5 (配重初始展开位)
  - mask: 放开自由腿 L_hip_roll(0) 给 RL (配重调节), 仍钉住 pitch/knee(1-2)+支撑腿(4-5)
  - 新增 _reward_counterweight: 自由腿 L_hip_roll 在有效配重区间[-1.2,0.5]给分
  - fold_pose 改为罚支撑腿不直 + 自由腿不微屈
修改文件:
  - src/unilab/envs/locomotion/xqrobotwl/single_leg.py (常量/mask/FF/reset/奖励)
  - conf/ppo/task/xqrobotwl_single_leg_flat/mujoco.yaml (counterweight: 2.0, wheel_off 8→6)
验证方法: smoke test — 机身+30°, 自由腿配重-0.5, 支撑腿伸直, L_hip_roll 给+2动作能移配重(+1.69)
评估结果: run12 (配重设计) 训练中
后续计划: monitor balance_complete → 确定性评估
关联日志: 01-05
---
