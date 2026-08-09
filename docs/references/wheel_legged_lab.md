# Wheel-Legged-Lab 参考

> 与本项目**同构的第二套实现**：同一款轮腿机器人,在 Isaac Lab 上的完整跳跃训练工程。
> 与本项目 (UniLab/MuJoCo) 的「Wheeled-SRL」(SLIP-FSM 前馈 + PPO) 属同一范式,可互相印证。

## 来源

- 路径: `/home/robot/xiaoq/projects/Wheel-Legged-Lab`
- 仓库: `https://github.com/zyicome/Wheel-Legged-Lab`
- 环境: NVIDIA Isaac Sim 5.1.0 / Isaac Lab 2.3.2 / RSL-RL 5.0.1 / PyTorch 2.7.0+CUDA 12.8
- 机器人模型来源: [clearlab-sustech/Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) (BSD-3-Clause),`wl_dealed.urdf` 将腿杆碰撞 mesh 替换为 box
- 参考学习对象: fan-ziqi/robot_lab (Isaac Lab Manager-Based 工程组织)
- 性质: 个人学习/复现项目,**未完成真实部署与系统 Sim-to-Real**

## 核心设计一句话

**VMC 虚拟腿控制 + 六阶段接触驱动跳跃状态机 + 阶段参考/策略残差动作混合 + 10 阶段一键流水线。**
策略输出左右虚拟腿角、虚拟腿长和轮速参考,由 VMC 映射到关节力矩;跳跃阶段由状态机提供腿长参考轨迹,PPO 只学残差。

## 机器人形态

两腿两轮开链轮腿平衡机器人,6 DOF:

| 索引 | 关节 | 类型 |
|------|------|------|
| 0 | `lf0_Joint` | 左髋 (revolute) |
| 1 | `lf1_Joint` | 左膝 |
| 2 | `l_wheel_Joint` | 左轮 |
| 3 | `rf0_Joint` | 右髋 |
| 4 | `rf1_Joint` | 右膝 |
| 5 | `r_wheel_Joint` | 右轮 |

### 运动学/物理参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 总质量 | 12.28 kg | 开环测试实测 |
| 大腿 `l1` / 小腿 `l2` | 0.15 / 0.25 m | |
| 髋偏移 `offset` | 0.054 m | |
| 腿长范围 `L0` | **0.18–0.30 m** | 虚拟腿长工作区间 |
| 默认腿长 `l0_offset` | 0.237 m | |
| 轮半径 | 0.0675 m | |
| 半轮距 | 0.25 m | |
| 腿关节力限 | 30 N·m | |
| 膝关节行程 | [-1.0, 1.25] rad | |

### 控制频率

| 参数 | 值 |
|------|-----|
| 控制频率 | 100 Hz (control dt=0.01s) |
| 物理频率 | 200 Hz (physics dt=0.005s, decimation=2) |

## 控制方法:VMC 虚拟模型控制

### 动作空间 (6 维)

```text
[theta0_ref_L, L0_ref_L, wheel_vel_L, theta0_ref_R, L0_ref_R, wheel_vel_R]
```

- `theta0`: 虚拟腿角度(相对垂直方向)
- `L0`: 虚拟腿长度
- 轮速参考经 **PI 积分**消除稳态误差(积分限幅防 windup)

### VMC 映射

正向运动学由两连杆 `theta1, theta2` 计算虚拟腿 `(L0, theta0)`,再经 VMC 雅可比把任务空间力/力矩映射到髋、膝关节力矩。轮速单独 PI 控制,轮电机力矩与腿力矩组合成 6 维关节力矩,受 `effort_limits` 裁剪。

### VMC 参数 (平地基础 `VmcActionsCfg`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `action_scale_theta` | 0.35 | 腿角动作缩放 |
| `action_scale_l0` | 0.06 | 腿长动作缩放 |
| `action_scale_vel` | 24.0 | 轮速动作缩放 (vx=1 时约需 21.8 rad/s) |
| `action_clip` | 1.0 | |
| `kp_theta` / `kd_theta` | 60 / 3 | 腿角 PD |
| `kp_l0` / `kd_l0` | 600 / 20 | 腿长 PD |
| `feedforward_force` | 60 N | 支撑前馈 |
| `kp_wheel` / `kp_wheel_integral` | 0.5 / 0.2 | 轮速 P / I |

> 跳跃阶段通过 `use_phase_dependent_gains` **按阶段改增益**:THRUST 时 `kp_l0×1.25`、前馈×1.15;LANDING 时 `kd×2.5`、前馈×1.20;FLIGHT 上升段收腿前馈≈0,预着陆再恢复支撑前馈。

## 跳跃任务设计

### 六阶段接触驱动状态机

```text
IDLE ──trigger+双轮接触+姿态/速度允许──▶ CROUCH ──L0≤0.22m且≥0.22s──▶ THRUST
  ▲                                          │
  │                                          ├─蹬伸≥0.18s、L0≥0.31m、vz≥0.4──▶ FLIGHT
  │                                          └─0.15s未离地──▶ RECOVERY(失败)
  └──────────── RECOVERY ◀── LANDING ◀── FLIGHT
                   ▲
                   ├─稳定0.5s且apex rise≥80%目标──▶ IDLE(成功)
                   └─1.5s未恢复──▶ IDLE(失败)
```

关键阈值 (Stage B2 默认):

| 参数 | 值 |
|------|-----|
| 接触力阈值 | 2 N |
| 离地/着地确认 | 连续 2 步 |
| CROUCH 参考腿长 / ready 阈值 | 0.19 / 0.22 m |
| CROUCH 最短 / 最长 | 0.22 / 0.42 s |
| THRUST 参考腿长 | 0.30 m |
| THRUST 最短 / 最长 | 0.18 / 0.32 s |
| 释放条件 | vz≥0.40 m/s 且 L0≥0.31 m |
| 确认起跳 | FLIGHT 中连续 2 步无接触且 vz≥0.10 |
| FLIGHT 收腿参考 | 0.18–0.19 m |
| LANDING 吸能压缩 | 0.21 m,高阻尼 |
| RECOVERY 稳定时间 | 0.50 s |
| 成功高度 | apex rise ≥ 80% × 命令目标 |
| 成功滞空 | ≥ 0.12 s |

### 跳跃命令 `jump_command = [trigger, target_height, target_distance]`

| 参数 | 值 |
|------|-----|
| 命令周期 | 3–4 s |
| 有跳跃周期比例 | 90% (保留无跳周期防移动遗忘) |
| 触发延迟 | 0.6–1.0 s |
| trigger 脉冲宽度 | 0.10 s |
| 目标上升高度 | B1: 0.04–0.06 → C1: 0.09–0.12 → C2+: 0.08–0.12 m |
| 目标距离 | 0 (目标落点/障碍物阶段再加) |
| 跳跃时速度 | 锁存 `vx` 为整跳速度目标,飞行中禁止重采样 |

### 观测:36/153 → 48/165

| 组 | 维度 | 构成 |
|----|------|------|
| **Actor (flat)** | 36 | base_lin_vel(3)+base_ang_vel(3)+projected_gravity(3)+velocity_cmds(3)+leg_joint_pos(4)+leg_joint_vel(4)+theta0(2)+theta0_dot(2)+L0(2)+L0_dot(2)+joint_wheel_vel(2)+last_action(6) |
| **Actor (jump)** | 48 | 36 + 12 (jump_phase 6×one-hot + wheel_contacts 2 + jump_command 3 + jump_phase_time 1) |
| **Critic (flat)** | 153 | 3+39+6+6+6+77+6+1+3+6+1+1 (含 action history×3、全关节、height_scan 77、torques、base_mass、base_com、default_joint_offset、摩擦/恢复) |
| **Critic (jump)** | 165 | 153 + 12 |
| **障碍物阶段 Actor** | 58 | 48 + 10 (前向高度扫描 10 bin) |

跳跃观测用 **one-hot 阶段编码**而非单个编号,避免网络误以为相邻数字代表相近物理状态。

### 分阶段跳跃奖励

RewardManager 会将权重乘 `step_dt=0.01`,事件奖励权重看似很大但单次贡献有限。

**继承平地并加阶段门控的项:**

| 奖励 | 权重 | 门控 |
|------|------|------|
| `lin_vel_z` | -0.5 | 仅 IDLE/LANDING/RECOVERY (避免"向上加速"与"禁止竖直速度"冲突) |
| `base_height` | +2.5 | 仅 IDLE/RECOVERY |
| `orientation` | -4.0 | 全程 |
| `dof_acc` | -5e-9 | 比平地弱,允许快速推蹬 |
| `torques` | -5e-5 | 弱 |
| `action_rate` / `action_smooth` | -0.001 / -0.001 | 弱 |
| `track_lin_vel` / `track_ang_vel` | +3.0 / +1.0 | 保留移动 |
| `termination` | -100 | 事件约 -1 |

**跳跃专用奖励:**

| 奖励 | 权重 | 阶段/事件 | 目的 |
|------|------|------|------|
| `jump_crouch` | +8 | CROUCH | 跟踪 0.22m 下蹲腿长 |
| `jump_phase_action` | +6 | CROUCH-THRUST-FLIGHT-LANDING | 腿长残差配合阶段参考 |
| `jump_thrust_pose` | +5 | THRUST | 引导伸向 0.30m |
| `jump_thrust_speed` | +5 | THRUST | 奖励腿长快速增长 |
| `jump_takeoff` | +8 | THRUST | 稠密正向 vz + 弹道目标 |
| `jump_takeoff_event` | +100 | 首次真离地 | 只奖励带正 vz 的确认离地 |
| `jump_height` | +6 | FLIGHT | 高度进展 + 目标 apex 跟踪 |
| `jump_airborne` | +2 | FLIGHT | 确认真离地 |
| `jump_symmetry` | +0.5 | CROUCH-LANDING | 左右腿协调 |
| `jump_landing_pose` | +1 | FLIGHT/LANDING | 准备缓冲腿长 |
| `jump_landing_soft` | +40 | 首次着地 | 小落地速度 + 小倾角 |
| `jump_landing_impact` | -15 | 首次着地 | 惩罚落地 vz² |
| `jump_recovery` | +1.5 | 真离地后 RECOVERY | 双轮着地 + 直立 |
| `jump_success` | +200 | 成功事件 | 单次约 +2 |
| `jump_failure` | -100 | 失败事件 | 单次约 -1 |

> 关键设计:成功高度要求 **apex rise ≥ 80% × 目标** — 只要短暂离地就给成功奖励会产生"极小弹跳"漏洞。

### 阶段参考 + 策略残差 (核心技巧)

```text
最终腿长动作 = 阶段参考动作 + residual_scale × 策略腿长动作
```

- Stage B1 用 `residual_scale=0.15`;进入 B2 后降到 `0.05`(防止旧策略残差抵消已验证的深蹲轨迹)
- **轮速和腿角仍完全由策略控制**
- 参考轨迹是**连续、随状态渐变**的(高度条件化下蹲、按 vz 分段预伸、落地压缩),不是 6 个固定动作向量
- 设计动机:从随机噪声中搜索一个仅约 0.2s 的蹬伸时序几乎不可能,给参考后 PPO 只学修正/姿态/轮速/落地/恢复

## 训练演进中的关键经验 (强烈建议论文引用)

### 第一轮训练停滞的四个相互强化问题
1. **假离地**:状态机在 CROUCH 就累计无接触步数,下蹲卸载被当成起跳(日志起跳 vz<0 为直接证据)→ 修复:每次进入新阶段清零该阶段接触计数,只有 FLIGHT 中连续两步无接触且 vz>0.10 才确认起跳
2. **奖励漏洞**:未真离地的 RECOVERY 仍拿恢复/对称奖励 → 关闭
3. **探索噪声发散**:两腿长动作 std 增长到 16–17,动作长期被裁剪到极限 → 训练期间约束 std∈[0.05, 0.50]
4. **搜索问题**:绝对动作策略需要偶然发现"蹲到位—0.2s 快速蹬伸—收腿"短时序 → 阶段参考+残差

### 平地 checkpoint 迁移陷阱
平地 `model_1999.pt` 两腿长动作 std≈19.69/20.26。确定性 play 没问题,但 PPO 采样时动作几乎总被裁剪到极限。`expand_rsl_checkpoint_for_jump.py` 转换时把 std 上限设为 0.30,重置为标准差:
`[0.1819, 0.3000, 0.2985, 0.1647, 0.3000, 0.3000]`

### 机身高度 ≠ 轮端净空
Stage C1 `J机身升≈0.11m` 但 `J轮高≈0.05m`:因为下降准备从 vz=+0.30 就开始,且 FLIGHT 保留 55% 支撑前馈,轮腿在到达 apex 前已重新伸长。→ 必须**显式测量两轮较低者的真实净空**并单独奖励。

### 航向冲突
C3-L1 中 `heading_command=False` + `wz∈[-0.20,0.20]` 让策略被角速度命令要求持续转向,同时 `track_heading` 又要求回到 yaw=0。→ 改为 `heading_command=True`,wz 由航向误差自动生成。

### 开环物理验证先行
两轮平衡机器人**不能**纯开环测跳跃(所有动作归零→倾倒,18 组有效跳跃 0 个)。平衡策略稳定 + 腿长开环 → 有效跳跃 13/18,最大上升 0.114m,最大 vz 1.27m/s。→ RL 前先验证 VMC/行程/力矩/接触是否具备起跳能力。

## 分阶段训练流水线

### 10 阶段

```text
flat → recovery → terrain_reactive → jump_flat → high_landing
     → clearance → moving_curriculum → target_landing
     → obstacle_oracle → obstacle_perceptive
```

### 自动验收机制
- 最近 20 iteration 滑动平均,连续 3 次通过
- 各阶段联合检查(速度/成功率/净空/柔落/力矩饱和 ≤0.05/恢复失败率),不只靠成功率尖峰
- 某阶段用尽 max_iteration 未通过 → 停止并保留 checkpoint

### Checkpoint 迁移
- 同维度:直接 `--load_weights_only` (新优化器,不延续旧 Adam 动量/迭代号)
- 跨维度:自动扩展 actor/critic 首层输入,新列随机初始化 (36→48→58)
- `terrain_reactive → jump_flat` 用 `expand_rsl_checkpoint_for_jump.py`

### 移动跳跃速度课程 (C3-Auto)
0.2→1.0 m/s 五档,每档统计窗口 ≥512 episode / ≥1024 次跳尝试,四项同时达标(移动得分≥0.78、跳成功≥0.75、柔落≥0.75、航向≥0.80)并连续两窗口通过才升级。

### 障碍物几何课程 (7 档)

| `O档` | 高度上限 | 宽度 | 前进速度范围 |
|---:|---:|---:|---:|
| 0 | 0.02 | 0.035 | 0.45–0.60 |
| 1 | 0.04 | 0.035 | 0.45–0.65 |
| 2 | 0.05 | 0.050 | 0.50–0.65 |
| 3 | 0.06 | 0.050 | 0.50–0.70 |
| 4 | 0.07 | 0.065 | 0.60–0.75 |
| 5 | 0.08 | 0.065 | 0.60–0.75 |
| 6 | 0.08 | 0.080 | 0.70–0.75 |

触发解析式:`target_distance = clamp(|vx|×0.18s, 0.08, 0.16)`,`trigger_distance = |vx|×0.44s + takeoff_standoff`。课程晋级门槛(课成功≥0.40、课跨越≥0.68、课碰撞≤0.38)与最终严格成功判定(`O成功`)分离。

### 上电安全状态机 (实机部署相关)
`PASSIVE → RAMP → EXTEND → STABILIZE → HANDOFF → COMPLETE`,VMC 动作项新增逐环境 `motor_enable_scale` 输出门,0 力矩时清零轮速 PI 积分防 windup。

## 关键参数表 (可直接对比)

| 参数 | Wheel-Legged-Lab | UniLab xqrobotwl jump (纯PPO) |
|------|------------------|------------------------------|
| 框架 | Isaac Lab + RSL-RL | MuJoCo 后端 + 自研 PPO |
| 动作维度 | 6 (VMC 虚拟腿) | 8 (6 关节角 + 2 轮速) |
| 控制频率 | 100 Hz | 100 Hz (ctrl_dt=0.01) |
| num_envs | 4096 | 1024 |
| 网络 | [256,128,64] | [512,512,256,128] |
| learning_rate | 3e-4 | 1e-4 |
| entropy_coef | 5e-4 (jump) | 5e-3 |
| 跳跃状态机 | 6 状态接触驱动 | SLIP-FSM 6 状态 (jump_srl) |
| 前馈方式 | 连续腿长参考 + 残差(0.05–0.15) | 固定 8D 动作前馈 ×0.15 + PPO |
| 观测 Actor | 36→48→58 | 297→315 |
| 成功判定 | 真离地+apex≥80%+滞空≥0.12s | jump_phase 门控 |
| 最终能力 | 0.08×0.08m 障碍跨越 | 平台 0.15m 跳跃 |

## 对 xqrobotwl 的借鉴要点

1. **VMC 虚拟腿控制**:输出虚拟腿角/腿长比直接输出关节角更利于"平衡"与"跳跃"解耦。xqrobotwl 目前是 6 关节角直接位置控制,可考虑引入虚拟腿中间表示。
2. **阶段参考 + 策略残差**:xqrobotwl 的 SRL 用固定 8D 前馈×0.15 已属此范式;本项目的**连续/状态条件化参考**(高度条件下蹲、按 vz 分段预伸、落地压缩)更精细,可直接升级 SRL 的 `_FSM_FEEDFORWARD` 为状态依赖的连续轨迹。
3. **接触驱动的成功判定**:用"真离地 + apex≥80% + 滞空≥0.12s + 软着陆 + 恢复"替代纯高度判定,能根治"极小弹跳"/假离地漏洞。xqrobotwl 的 `jump_height` 已加 `knee_ok` 门控,但尚无 min_air_time 硬约束。
4. **机身高度 ≠ 轮端净空**:跳跃评估应同时看机身上升和轮端真实净空,后者才代表越障能力。xqrobotwl 的 platform 场景可用此指标增强。
5. **开环物理验证先行**:RL 前先用平衡策略稳定 + 腿长开环扫描验证 VMC/力矩/接触能否离地,避免把物理不可行的任务扔给 PPO。
6. **探索噪声约束**:绝对动作策略的腿长/关节 std 易发散(16–17),训练期间应设 std 上下限 [0.05, 0.50]。
7. **分阶段迁移训练**:平地→跳跃用 checkpoint 首层扩展 + `load_weights_only`,每阶段新优化器;课程晋级用多指标联合窗口而非单一成功率。
8. **上电安全**:实机部署前应加 `motor_enable_scale` 输出门和力矩渐入状态机,避免启动瞬间全力矩。

## 已发布资产 (可直接参考/对比)

- 预训练模型 `checkpoints/wheel_legged_moving_jump_model_844.pt` (环境 `Wheel-Legged-Jump-Moving-Curriculum-Flat-v0`,1.0 m/s / 1.2 rad/s)
- TensorBoard 曲线 `docs/tensorboard/model_844/`
- 演示 GIF/MP4:跨越 7cm 障碍、单/多环境移动跳跃、键盘控制
