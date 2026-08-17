# [NN] jump_srl_vmc (SLIP-FSM+VMC+PPO) §7.5 改进 — 落地恢复 + 阶段映射修复 + 直立奖励

**日期**: 2026-08-15
**来源**: 任务指定 — `XqRobotWLJumpSRLVMC` 向 §7.5 平地跳跃验收目标靠近 (成功率 ≥90%)
**任务**: `XqRobotWLJumpSRLVMC` / `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml`
**关联**: [[26_final_compare_improved_vmc]], [[20_vmc_slip_refactor_ppo_landing]], [[25_jump_weakness_diag_fix]]

---

## 基线复现 (verify_jump.py 实测)

```
uv run python tools/xqrobotwl/verify_jump.py --task XqRobotWLJumpSRLVMC \
  --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/2026-08-14_14-18-36_mujoco/model_9999.pt --steps 600
→ task=XqRobotWLJumpSRLVMC survived=600/600 terminated=False max_base_z=0.964 standing_z=0.476 jump_height=0.488 air_frac=0.14
```

- **方差大**: 3 次运行 2 次 600/600, 1 次 327/600 终止。8-episode 详细诊断: **6/8 失败** (75% 失败率)。
- 失败率 ~60-75%, 距离 ≥90% 成功率差很远。
- 跳得不错 (0.49m, air 14%), 空中收腿 OK。问题集中在**落地与落地后恢复**。

## 诊断 (数据驱动, diag_landing 8-episode)

失败模式三连 (共同杀手 = **俯仰角 tilt 涨过 45° 终止**):

1. **蹬伸期俯仰过大** (EP5): thrust 相 (fsm 1) 双轮着地, 膝盖过度伸展到 -1.15/+1.07, tilt 5.9°→43°, 未离地就倾倒。
2. **落地硬冲击 + 深塌** (EP4/7): 落地 base_z 塌到 0.27-0.28, tilt 在 recovery 中持续涨到 >44°。
3. **落地后缓慢塌缩** (EP6): fsm 回 idle 后, 双轮仍着地 (1,1), 但膝盖持续弯曲 (-0.33→-0.71), base_z 缓慢下沉 0.44→0.39, tilt 23°→42.6°, 约 0.3s 内慢慢摔倒。

**根因分析**:

| # | 根因 | 证据 |
|---|------|------|
| a | **FSM 阶段映射错位**: SLIP-FSM 六状态 (-1 idle, 0 crouch, 1 thrust, 2 flight, 3 prelanding, 4 landing_absorb), 但 `_jump_leg_reference` 把 state 3 当"落地压缩"(0.42→0.30), state 4 用默认 l0_offset。落地缓冲增益 (landing_kd/ff_scale) 也绑在 state 3 → **实际触地冲击发生在 state 4, 却用普通增益, 冲击不吸收** | 代码审查 `jump_vmc.py:127-137, 189-191` |
| b | **无落地恢复奖励**: SRLVMC 用 SRL 奖励集, 无 `landing_recovery`。落地后没有强激励恢复直立 → 学不到"跳完站稳" | 训练日志 `reward/landing_soft≈0.1, 无 landing_recovery` |
| c | **landing_soft 门控错**: 绑 `jump_phase>=35` (触发窗口内), 落地时 trigger 已关 phase 归 0 → 落地缓冲奖励基本为 0 | `reward/landing_soft≈0.1` |
| d | **蹬伸俯仰无抑制**: `orientation -5.0` 太弱 (tilt 30° 只 -1.25/step), vertical_thrust (~8.6) 主导 → 策略为跳高容忍俯仰 | EP5 thrust tilt 43° |
| e | **wheel_ground_matching 是死奖励**: config 有 scale 20 但 SRLVMC 无该 reward fn → 未激活, 轮地匹配/防打滑不学 | 代码审查 |

## 修改 (只改 SRLVMC 隔离代码 + 本任务 config)

### 代码 `src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py`

1. **FSM 阶段映射修复** (override `_jump_leg_reference` + `get_l0_control_parameters`):
   - state 3 (prelanding) **保持伸腿 0.42** 准备触地 (不再提前压缩)
   - state 4 (landing_absorb) **落地缓冲压缩 0.42→0.30**, 高阻尼 + 高前馈吸收冲击
   - landing 增益 (kd×3.0, ff×1.30) 绑到 **state 4** (实际冲击), state 3 只加轻阻尼
2. **新增 `_reward_landing_recovery`**: 真实腾空→落地后窗口内 (landing_timer>0), 双轮着地 + 直立 (tilt exp) + 高度接近 target → 驱动落地后恢复站立。scale 12.0
3. **新增 `_reward_jump_upright`**: 跳跃窗口内 (phase≥1) 奖直立 (tilt exp, σ=0.2), 抑制蹬伸/腾空/落地俯仰。scale 12.0
4. **重写 `_reward_landing_soft`**: 门控 `landing_timer` (真实落地) 而非 phase>=35。scale 4.0
5. **新增 `_reward_wheel_ground_matching`**: fsm 1/3/4 内轮地速度匹配 (wheel_vel×sign×R vs 前向速度), 防打滑。scale 20.0 (原 config 就有, 现在真正激活)
6. **注入 info**: `_compute_reward` 把 `fsm_state` + `wheel_vel` 写入 info 供新奖励用
7. **cfg 新增 `landing_window: int = 50`**, env `__init__` 用它覆盖 base 的 10 → 落地缓冲奖励窗口 0.5s
8. **landing_recovery 门控修正**: 初版 gate 在 `landing_timer>0`, 但 eval 的 trigger 脉冲在落地后即关 → `_update_jump_air_progress` 里 `idle>0` 会把 landing_timer 清零 → 恢复奖励 eval 时完全不激活。改为**常开** (同 pure PPO), 奖"双轮着地+直立+高度接近 target"姿态; FSM 参考仍强制触发时蹲-蹬, 不可刷站立白拿

### 配置 `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml`

| 参数 | 旧 | 新 | 理由 |
|------|----|----|------|
| `env.landing_window` | — | 50 | 恢复窗口 0.5s (原 10 步太短) |
| `vmc.prelanding_start_vz` | 0.0 | 0.2 | 提前一点伸腿准备落地 |
| `vmc.landing_compression_time` | 0.18 | 0.12 | 压缩在 state4 的 0.15s 窗口内完成 |
| `vmc.landing_kd_scale` | 2.5 | 3.0 | 落地吸收更高阻尼 |
| `vmc.landing_ff_scale` | 1.20 | 1.30 | 落地吸收更高前馈支撑 |
| `vmc.thrust_kp_scale` | 3.0 | 2.5 | 蹬伸略收敛 (防俯仰) — 现网模型实测 3/5→1/5 失败 |
| `vmc.thrust_ff_scale` | 3.0 | 2.0 | 蹬伸前馈略降 (防俯仰) |
| `vmc.thrust_kd_scale` | 0.25 | 0.5 | 蹬伸阻尼略增 (防过爆俯仰) |
| `reward.scales.landing_recovery` | — | 12.0 | 新增落地恢复奖励 |
| `reward.scales.jump_upright` | — | 12.0 | 新增跳跃直立奖励 |

## 训练后效果 (待训练完成补充)

## 参数调整好坏 (待评估补充)

## 验证方法

- 冒烟测试: env 构建/step 正常, 新奖励函数激活
- 短训练 4000 iter 快速验证 → 每 1000 iter verify_jump.py
- 最终: `verify_jump_repeat.py --cycles 10` 统计 §7.5 成功率 (launch+landed+recovered+survived)
- 阶段姿态 + 轮速匹配检查

## 后续计划

- 4000 iter 有效 → 全量 10000 iter
- 无效 → 回调 (landing_recovery 权重 / thrust 强度 / 参数)

## 文件清单

- `src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py` (FSM 映射 + 新增奖励)
- `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml` (landing 参数 + 奖励 scale)
