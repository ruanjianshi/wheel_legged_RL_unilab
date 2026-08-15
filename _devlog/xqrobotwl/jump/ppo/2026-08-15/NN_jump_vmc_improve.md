# NN PPO+VMC 跳跃改进: 堵住"下蹲相慢伸白拿 launch_rise" exploit, 逼出真实蹬伸

**日期**: 2026-08-15
**来源**: 独立任务 `xqrobotwl_jump_vmc_flat` (XqRobotWLJumpVMC) — 基线对称几何重训
  model_9999 实测 `survived=600/600 jump_height=0.262 air_frac=0.02`, 目标是 §7.5
  平地跳跃: 真实阶段链(下蹲→上跳→收腿→落地)+轮速匹配+稳定落地+成功率≥90%。
**关联**: [20_vmc_slip_refactor_ppo_landing](../../../2026-08-06/20_vmc_slip_refactor_ppo_landing.md),
  [17_vmc_jump_param_fix](../../../2026-08-06/17_vmc_jump_param_fix.md),
  [01_phase_gated_rewards](../../../2026-07-14/01_phase_gated_rewards.md)

---

## 基线复现 (verify_jump.py 实测)

```
task=XqRobotWLJumpVMC survived=600/600 terminated=False max_base_z=0.752 standing_z=0.490 jump_height=0.262 air_frac=0.02
```

## 诊断 (数据驱动)

1. **diag_jump_trajectory.py + 深度 VMC 诊断** 显示: FSM 下蹲相(fsm=0, trigger 开
   后 ~35 步)内策略把 L0 从 0.29 慢慢伸到 0.56, **轮子全程接地 (wl=wr=1)**;
   FSM 进入蹬伸相(fsm=1)时腿已完全伸直, 蹬伸无力 → 几乎没有真正腾空。
2. **训练 reward log 最后 10 iter**:
   ```
   reward/launch_rise: 4.52   (scale 40 → 慢伸白拿, 主导项)
   reward/landing_soft: 2.06
   reward/wheel_air_time: 0.29 (scale 20 → air_factor 仅 0.015)
   reward/jump_height: 0.007   (scale 12, 门控 air → 几乎拿不到)
   reward/crouch_depth: 0.023  reward/crouch_prep: 0.031
   ```
   即策略靠 "下蹲相缓慢伸腿" 赚 launch_rise + landing_soft, 从不为真实腾空负责。
   `jump_height` 被 `air_factor = 1 - wheel_contact` 门控, 轮不离地 → 永不激活。
3. **物理对照**: SRL+VMC (residual, feedback_gain=0.5, 同一 SLIP-FSM 参考) 实测
   `jump_height=0.455 air_frac=0.14`, 证明参考轨迹本身能产生真实跳跃;
   **PPO+VMC full-action (fb=1.0) 让策略能完全盖过参考, 是差距根源**。
4. 膝关节力矩在下蹲深位饱和 ±50 N·m, 深蹲到 base_z≈0.40 (接近 crouch_height_target)。

## 根因

PPO+VMC 是 **full-action 参考叠加** (`final_L0 = 参考 + 1.0×策略动作`)。策略保留
全部权限, 学会了在下蹲相慢慢伸腿——物理上这是"准静态抬升", 轮子不离地; 但
`launch_rise` (scale 40) 奖励任何 trigger 窗口内接地的上升, 慢伸正好刷满它。
`jump_height` 又门控在 air_factor, 于是策略对真实跳跃零回报 → 困在假跳高解。

## 修改

### 配置 `conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml`
| 参数 | 旧 | 新 | 理由 |
|------|-----|-----|------|
| `reward.jump_height` | 12 | **15** | 对齐四版本中其他版本, 加大真实腾空回报 |
| `reward.launch_rise` | 40 | **35** | 已门控到蹬伸相, 权重略降避免过度主导 |
| `reward.anti_early_extend` | 无 | **8.0** | 新增: 下蹲相提前伸腿惩罚 |
| `vmc.thrust_ff_scale` | 3.0 | **3.5** | 更强蹬伸前馈 (110×3.5=385N/腿) |
| `vmc.thrust_kd_scale` | 0.25 | **0.15** | 更低腿长阻尼 → 爆发式伸展 |
| `vmc.feedback_gain` | (默认1.0) | **0.7** | full-action→参考主导混合 |
| `vmc.crouch_upper_bound` | 无 | **0.38** | anti_early_extend 的 L0 上界 |

### 代码 `src/unilab/envs/locomotion/xqrobotwl/jump_vmc.py` (任务隔离)
- `step()`: `actions[:,2/5] = target_action + fb×policy`。fb=0.7 使 SLIP-FSM 参考
  主导蹲→蹬时序, 策略保留大部分权限但无法完全取消参考 (SRL+VMC 用 0.5 已验证)。
- 新增 `_reward_launch_rise` 覆写: 门控到 **FSM 蹬伸相 (fsm_state==1) + 接地**。
  下蹲相慢伸不再计分 → 策略必须等参考切到蹬伸相才拿升高奖励。
- 新增 `_reward_anti_early_extend`: fsm_state==0 且 trigger 开时, L0 超过
  `crouch_upper_bound` (0.38) 惩罚, 直接惩罚诊断到的提前伸腿行为。
- `_init_reward_functions()` 覆写注册 `anti_early_extend`。

### 代码 `src/unilab/envs/locomotion/xqrobotwl/vmc.py` (可选新字段, 默认不影响其他)
- `XqRobotWLVMCConfig` 增加 `feedback_gain=1.0`, `crouch_upper_bound=0.38`。
  SRL+VMC 从 `reward_config.feedback_gain` 读取, 不受影响。

## 训练后效果 (训练中, 待评估补充)

- 启动后早期 reward log: `launch_rise≈0.001` (门控生效), `jump_height≈0.06-0.09`,
  `wheel_air_time≈0.03-0.05`。需等 3000-10000 iter 评估。

## 参数调整好坏

| 改动 | 预期 | 坏处/风险 |
|------|------|-----------|
| fb=0.7 参考主导 | 逼出真实蹬伸时序 | 策略跳高上限略降 (无法完全加码) |
| launch_rise 门控蹬伸相 | 堵住慢伸 exploit | 若策略仍不会等蹬伸相, 训练更慢 |
| anti_early_extend 8.0 | 直接惩罚提前伸腿 | 惩罚过强可能压制下蹲后的一点弹性 |
| thrust_ff 3.5 / kd 0.15 | 更强爆发蹬伸 | 落地下冲更大, 需盯存活率 |

## 验证方法

- 训练中每 1000 iter 用 `tools/xqrobotwl/verify_jump.py` 测跳高/air_frac/survived
- 最终 `--steps 2200+` 重复跳 ≥10 次统计成功率 (§7.5: ≥90%)
- `tools/xqrobotwl/record_jump_trajectory.py` + `plot_jump_trajectory.py` 检查
  下蹲→上跳→收腿→落地 完整阶段链
- `tools/xqrobotwl/compare_jump.py` 与其余三算法对比

## 后续计划

- [ ] 等训练 3000 iter 评估, 若 air_frac 仍低 → 加 takeoff 奖励或调 fsm 时序
- [ ] 若跳高了但落地存活差 → 调 landing_soft/landing_kd_scale
- [ ] 达标后渲染视频 + 版本备份 (backup/XqRobotWLJumpVMC_*)

## 关联日志

- [NN_jump_srl_improve](NN_jump_srl_improve.md) (同日 SRL 任务, 独立)
