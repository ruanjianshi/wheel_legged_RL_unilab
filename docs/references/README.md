# 参考项目

## 目录

| 项目 | 日期 | 说明 |
|------|------|------|
| [CJ-003](cj003.md) | 2026-06-26 | 轮腿双足 8DOF, Genesis 框架, XqRobotV2 直接设计来源 |
| [HumanoidSW2](humanoid_sw2.md) | 2026-06-28 | 12DOF 双足, 正弦轨迹跟踪步态, 手臂平台 |
| [PAI 12DOF](pai_12dof.md) | 2026-06-29 | 12DOF 纯下肢双足, 内置蹲姿, 保守超参 |
| [WTW-NEW / Go1](wtw_new.md) | 2026-06-29 | 12DOF 四足, AER能量正则, RMA适应, 步态条件化 |
| [Wheel-Legged-Lab](wheel_legged_lab.md) | 2026-08-06 | 轮腿双足, Isaac Lab, VMC虚拟腿+六状态跳跃FSM, 阶段参考/策略残差, 10阶段流水线 |
| [跳跃站立稳定 RL](2026-08-16_jump_standing_stabilization_rl.md) | 2026-08-16 | P3 站立振荡: 站立显式奖励 + 动作LPF + smoothing + gait-conditioned 奖励路由 |
| [跳跃姿态 + 膝限约束 VMC](2026-08-16_jump_posture_knee_limit_vmc.md) | 2026-08-16 | P4 髋外展/无下蹲 + 膝过伸: 侧向对称姿态奖励 + 反向下蹲(CMJ) + L0膝限裁剪 + CaT |

## 规范

每个参考项目的文档应包含：
1. 项目基本信息（框架、方法、论文链接）
2. 机器人形态（DOF、关节布局）
3. 控制方法（动作空间、PD 参数、控制频率）
4. 训练设计（奖励函数、命令系统、课程学习）
5. 关键参数表（可直接对比）
6. 对 XqRobotV2 的借鉴要点
