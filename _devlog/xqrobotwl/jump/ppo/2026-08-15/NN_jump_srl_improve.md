# [NN] jump_srl (SLIP-FSM+PPO) §7.5 改进 — 落地恢复/轮速匹配/漂移

**日期**: 2026-08-15
**来源**: 任务指定 — jump_srl 向 §7.5 验收目标靠近
**任务**: `XqRobotWLJumpSRLFlat` / `conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml`
**关联**: 基线 = 对称几何重训 `2026-08-14_14-17-30_mujoco/model_9999.pt`

---

## 基线复现

```
uv run tools/xqrobotwl/verify_jump.py --task XqRobotWLJumpSRLFlat \
  --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-14_14-17-30_mujoco/model_9999.pt
→ task=XqRobotWLJumpSRLFlat survived=600/600 terminated=False max_base_z=1.154 standing_z=0.494 jump_height=0.661 air_frac=0.19
```

- 四版本中跳得最好 (0.60~0.66 m, air 19%), 存活 100%。
- 阶段链 (diag_jump_trajectory): 下蹲(0.55→0.31)→蹬伸→腾空(0.99~1.15m)→落地→恢复, 完整。
- **问题**: 落地后 base_z 塌到 0.27, 恢复站立慢且晃动; 触发关闭站立高度 ~0.48 (略低于 target 0.55); 水平漂移大 (eval_posture 报 3.36m)。

## 诊断 (根因)

1. **`wheel_ground_matching` 奖励从未生效 (bug)**: config `scales` 键是
   `wheel_ground_matching`, 但 `jump_srl.py` 的 `_reward_fns` 注册键是
   `wheel_air_time`。`run_reward_dispatch` 按 config 键查找 fn, 找不到即跳过 →
   该奖励全程为 0, 且 `wheel_air_time` fn 也因 config 无此键而失效。轮速匹配完全未训练。
2. **`action_magnitude` 奖励从未生效 (bug)**: config 有 `action_magnitude: -0.1`,
   但 `_reward_fns` 无此 fn → 静默跳过。
3. **landing_soft 门控弱**: 用 `phase >= 35` (触发窗口内无界增长), 落地缓冲在
   腾空/飞行阶段也发放; 且 scale 仅 4.0。落地后恢复站立的专项奖励完全缺失
   (无 landing_recovery)。
4. **落地塌陷**: diag 显示落地瞬间 base_z 降到 0.27 (腿屈曲到底), 恢复窗口约 0.2-0.3s,
   期间晃动。这是 §7.5 "稳定落地+恢复站立" 的最大威胁, 也是成功率的瓶颈。
5. **漂移**: 落地塌陷反弹 + 无轮速匹配 → 前冲动量未吸收, 站立期持续滑动。
6. eval_posture 对 jump 任务的 trigger 恒为 0.518 (>0.5) → 不是真实站姿测量,
   "z=0.712 站姿异常偏高" 是持续跳跃周期的平均高度, 非站立高度 (实际站立 ~0.49)。

## 修改方案

### 配置 `conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml`
- `landing_soft`: 4.0 → 8.0 (门控到真实落地)
- 新增 `landing_recovery: 8.0` — 落地后恢复稳定站立
- 新增 `anti_drift: -4.0` — 站立期水平漂移惩罚
- 新增 `landing_window: 30`, `min_grounded_steps: 5` — 落地恢复窗口
- `wheel_ground_matching: 20.0` (保留, 修复 fn 对齐后生效)

### 环境 `src/unilab/envs/locomotion/xqrobotwl/jump_srl.py`
- `XqRobotWLJumpRewardConfig` 新增 `landing_window` / `min_grounded_steps`
- 新增奖励 fn:
  - `_reward_landing_recovery`: 双轮着地+直立+高度接近目标, 门控 `landing_timer`
  - `_reward_anti_drift`: 触发关闭时惩罚水平速度超出 vx 指令的残余漂移
  - `_reward_action_magnitude`: 修复 config 键无 fn 的静默失效
  - `_reward_wheel_ground_matching`: 落地(FSM 3/4)轮速与地面速度匹配 + 飞行(FSM 2)轮不空转
- `_reward_landing_soft` 改为门控 `landing_timer` (真实腾空→落地), 无 `landing_timer`
  时回退旧 `phase>=35` (兼容 jump_srl_vmc)
- 新增 `_update_jump_air_progress`: 追踪真实腾空→落地 (had_air / landing_timer /
  grounded_steps), 排除 reset 初始自由落体
- `update_state` 补充 `fsm_state` / `wheel_vel` 到 info, 调用落地追踪
- `_init_reward_functions` 注册新 fn

## 基线重复跳评估 (eval_jump_repeat.py 新增)

```
attempts=16 (2ep×8跳)  airborne=15/16   recovered(恢复站立)=15/16  terminated=1/16
成功恢复率 = 0.94   (空中跳恢复率 = 1.00)
跳高 = 0.601 ± 0.058 m
恢复窗口漂移 = 0.492 ± 0.427 m  (单跳最大 1.48m)
空中轮速峰值 = 58.5 rad/s (≈ 3.80 m/s)  ← 严重空转
```

**结论**: 基线"恢复站立"成功率已达 0.94 (临门 §7.5 ≥90%), 但两个硬伤:
1. **空中轮速空转 58 rad/s** (3.8 m/s) — 落地必然打滑, 违反 §7.5 轮速匹配。
2. **恢复窗口漂移 0.49±0.43 m** (部分跳 1.4m) — 微动平衡不达标。
3. 1/16 跳落地跌倒终止 — 恢复稳健性不足。

## 关键修复 (reward 爆炸 → 有界)

初版 wheel_ground_matching 用原始角速度平方 (`sum(wheel_vel²)`), 基线空中 58 rad/s
→ 3364, ×scale 20 → **-852/step**, resume 后 mean reward -166, 会压垮训练。
改为:
- 轮速换算线速度 (rad/s × WHEEL_R → m/s)
- contact_error 截断 [0, 4.0] (≈2 m/s 失配), spin 截断 [0, 2.0] (≈1.4 m/s)
- `wheel_ground_matching` scale 20 → 8 (该奖励基线从未生效, 降幅避免初次激活过冲)
实测基线策略下 wgm 贡献最大 -32/step (而非 -852)。

## 第二处 bug: landing_recovery 的 upright 恒 0

`_reward_landing_recovery` 用 `arccos(clip(-gravity[:,2]))` 算 tilt, 但 jump_srl 的
gravity 来自 `upvector` 传感器 (直立时 +1), 负号使 upright 恒 0 → landing_recovery
永远不发。修复: 用正值 `arccos(clip(gravity[:,2]))`。修复后 landing_recovery 正常
发放 (训练 log: 0.064 → 0.198)。

另注: eval_jump_repeat 的 100-on/100-off 脉冲模式下, 落地恰好发生在 trigger-off
边界, landing_timer 在 idle 时清零, 故 eval 中 landing_soft/landing_recovery 不发
(只影响奖励数值, 不影响策略行为)。训练用 4s 命令窗口, 落地在 on 窗口内, 正常发放。

## 训练 (resume 微调)

- 新奖励 (landing_recovery/anti_drift/wgm/action_magnitude) 从零训收敛慢,
  改为 **从基线 model_9999 resume** (已会跳 0.6m), 只学新增行为。
- `algo.load_run=2026-08-14_14-17-30_mujoco algo.checkpoint=9999 algo.max_iterations=15000`
- 新 run: `2026-08-15_13-19-26_mujoco_resume_v2`, checkpoint 从 model_10000 起
- `logs/train/jump_srl_v2.log`
- (废弃 resume_v1 run 目录已删除: 2026-08-15_13-13-11_mujoco_resume_v1)

## 训练动态

| 阶段 | mean_reward | 说明 |
|------|------------|------|
| 初版 wgm -852 | -166 | 惩罚爆炸 (已修: 截断 + 降幅) |
| resume_v2 iter 10001 | -18~-24 | landing_recovery 修复后初始冲击 |
| iter 10047 | -6.97 | episode 697, 恢复中 |
| iter 10062 | +9.34 | episode ~790, 转正 |

- model_10000 (1 iter 微调后) verify: survived 171/600 — 首步 PPO update 冲击, 后期恢复。
- 趋势: mean_reward 转正, episode 增长 → 微调在学新奖励。

## 评估 (待补充)

- verify_jump / eval_jump_repeat (重复 ≥10 跳统计成功率) / eval_posture 复查

## 结论 (待补充)
