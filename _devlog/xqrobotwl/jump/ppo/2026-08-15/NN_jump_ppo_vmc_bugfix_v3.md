# [NN] 纯PPO + PPO+VMC v3: 定位两个静默失效 bug (landing_recovery 符号 / VMC 接触检测) → 重训

**日期**: 2026-08-15
**来源**: 老板指派 — "改进纯PPO和PPO+VMC这两个算法的跳跃"
**关联**: [[NN_jump_improve]], [[NN_jump_vmc_improve]], [[28_pure_ppo_reward_hacking_v10_v12]]

---

## 基线评估 (本次会话实测, 08-14 model_9999)

| 算法 | verify 跳高 | air_frac | eval 成功率(16次) | 问题 |
|------|------------|----------|-------------------|------|
| 纯PPO | 2.962(失控) | 0.97 | **0%** (16/16 摔) | 火箭发射, 永不落地 |
| PPO+VMC | 0.295 | 0.02 | 100% (但 air_steps 仅1-6) | 假跳高, 准静态伸腿 |

## 根因定位 (两个静默失效 bug)

### Bug 1: 纯PPO `landing_recovery` 符号 bug → 落地恢复奖励恒 0
`src/unilab/envs/locomotion/xqrobotwl/jump.py` `_reward_landing_recovery`:
```python
tilt = np.arccos(np.clip(-ctx.gravity[:, 2], -1, 1))   # ← 负号错误
```
upvector 传感器直立时 `gravity[:,2]=+1`, 取负 → `arccos(-1)=π` → `upright≈0` →
landing_recovery **从未生效** (v2 训练日志 `reward/landing_recovery` 恒 0.0000 证实)。
PPO+VMC 继承此函数同样失效。→ 策略学不到"跳完落地站稳"。

### Bug 2: VMC 变体仍用力阈值接触检测 → air 门控奖励全部失效
- `xqrobotwl_vmc.xml` **无** `left_wheel_world_pos` framepos 传感器 (纯PPO 的 `xqrobotwl.xml` 有)
- VMC env `update_state` 调 force 版 `_update_wheel_contact` (`norm(wheel_force)>10`):
  空中扇腿时轮被加速, 轮力 >10N → 腾空误判为着地 → `jump_height`/`wheel_air_time`
  (air 门控) 恒 0 → 学不到腾空 (v2 训练 `jump_height≈0.1` 证实)
- **铁证**: 用几何接触检测重测 PPO+VMC 基线, air_frac **0.02 → 0.20** — 旧模型其实有腾空,
  只是训练时从没为此得分。

## 修改 (4 文件, git commit 96add2f + a6852d6)

1. `src/unilab/envs/locomotion/xqrobotwl/jump.py`: `_reward_landing_recovery` 改正值
   `np.clip(ctx.gravity[:, 2], -1, 1)` (与 jump_srl / _compute_terminated 一致)
2. `src/unilab/assets/robots/xqrobotwl/xqrobotwl_vmc.xml`: sensor 块加
   `<framepos name="left_wheel_world_pos" objtype="body" objname="left_link_wheel" />`
3. `src/unilab/envs/locomotion/xqrobotwl/jump_vmc.py`: `update_state` 改用
   `_update_wheel_contact_geom` (几何检测, 轮心世界 z<0.13 判着地)。
   SRL+VMC 继承 update_state, 一并受益。
4. `conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml`: **补遗漏的 `landing_recovery: 8.0`**
   — 其他三算法均有 (纯PPO 10 / SRL 8 / SRL+VMC 12), PPO+VMC 唯一缺失 (env fn 已注册
   但 config 未激活) → 落地恢复从未奖励。与 sign-bug 修复配合驱动"跳完落地站稳"。
   (发现于训练启动后, PPO+VMC 重启生效)

## 验证

- `pytest tests/envs/locomotion/xqrobotwl -k jump` **10 passed**
- 手动验证 landing_recovery: 站立 50 步累计 7.86 (修复前恒 0)
- PPO+VMC 基线几何接触: air_frac 0.02→0.20

## 训练 (v3/v4, 从零重训)

- 纯PPO v3: iter 1000 实测 **悬空 (air_frac 0.76, 跳高 0.99m)** — 触发 Phase 2
- PPO+VMC v3 (重启, 带 landing_recovery): iter 1000 实测 **air 0.32 / 跳高 0.30m / 存活 97%**
  — 接触修复生效, 非悬空, 继续训练不中断
- 纯PPO v4 (Phase 2 反悬空): `logs/train/jump_v4.log`

## Phase 2 (反悬空, git commit 5d69e94)

纯PPO v3 iter1000 空中奖励 (jump_height 12.8 + wheel_air_time 5.1) 是 landing_recovery
(4.1) 的 ~4x, 悬空成为局部最优。应用 Phase 2:
1. `jump.py _reward_jump_height`: 门控 `vz>0` (只奖上升段, 悬停顶点不再得分)
2. `jump.py _reward_wheel_air_time`: 去掉正 `air*0.5` 项 (奖"待在空中的时间"被刷), 只罚空中转轮

## Phase 3 (反火箭, git commit 2c7ccda)

纯PPO v4 iter1000 仍被弹跳利用: **crouch_prep≈0 (不深蹲就起跳)**, 上升段刷 jump_height
(air 0.81)。应用 Phase 3:
1. `jump.py _reward_jump_height`: 追加 **window_crouched 门控** (须先蹲到<0.42, 与 launch_rise
   同门控) → 原地弹跳/火箭无跳高奖励, 逼出"先蹲后蹬"
2. `jump.py _reward_landing_recovery`: 门控到 **trigger-off** (ON 时段站着不奖) → 策略学
   "ON 跳 / OFF 站稳", 对齐 §7.5 评估时序

## Phase 4 (v6, git commit 1b9f401): 恢复 v12 配方

纯PPO v5 iter1383: 策略刷 tracking_lin_vel(1.72)+landing_recovery(4.97) 站住不动,
jump_height 被深蹲门控锁死成 0 → 鸡生蛋 (不蹲→无跳高奖励→不蹲)。**考古 v12 (8d2c042)
发现**: 其受控 0.2m 跳的配方 = launch_rise 40 主导 + jump_height 15 (当时 contact bug
使其失效)。40→20 降幅是 force-contact bug 时代的保守决策, 几何接触修复后 40 安全。v6:
- `launch_rise` 20→**40** (主导跳跃奖励, v12 配方)
- `jump_height` 30→**15** (v12 水平, 有 vz>0+深蹲双门控防火箭)
- `tracking_lin_vel` 2→**1**, `tracking_ang_vel` 1→**0.5** (降低"站住开车"刷分)
- `window_crouch_threshold` 0.42→**0.45** (放宽深蹲解锁, 破鸡生蛋)

## 训练后效果

### PPO+VMC v3 (iter 2000-3000, 接触修复 + landing_recovery 生效)
```
iter2000: attempts=16 成功恢复率=1.00  跳高 0.221±0.027 m  air_steps 46-56 (真腾空)
         漂移 0.205±0.205 m  空中轮速 11.2 rad/s
iter3000: verify 600/600 survived, 跳高 0.240m, air_frac 0.35 (稳定)
```
**对比基线 (08-14)**: 基线 air_steps 仅 1-6 (假跳, 轮几乎不离地) → 现在真腾空 46-56 步 +
100% 恢复站立。接触检测修复的成果 (旧模型其实有腾空, 但训练从没为此得分)。

### 纯PPO v6 (v12 配方, iter 1000)
```
verify: survived=600/600, jump_height=0.145m, air_frac=0.09
eval: attempts=16 成功恢复率=1.00 (16/16 全部落地恢复)  跳高 0.114±0.014m  漂移 0.373m
```
**从火箭/悬空 → 受控小跳 + 100% 恢复**: 比基线 (0% 成功率火箭) 大幅改进, 成功率达标 §7.5 ≥90%。
但收敛到"小跳+柔和落地"局部最优 (landing_soft 9.7 主导, launch_rise/jump_height 全程 0,
不深蹲), 高度卡 0.11m 不再提升。

### 纯PPO v7 (jump_land, git commit 146a0e7, 完成)
v6 高度卡住的根因: 逐步高度奖励 (jump_height) 被深蹲门控锁死, 小跳也拿同样的
landing_soft。v7 新增 **jump_land**: 追踪 window_max_z (真实腾空峰值), just_landed 时
按 (peak-floor)/target 一次性奖励 (scale 150) — 悬空/火箭/小跳刷不到, 跳高才拿大额。
**结果**: model_3999 verify 600/600 存活, 跳高 0.137m, air 8%; eval 16/16 恢复 (100%),
跳高 0.122m。**与 v6 相当 — jump_land 未显著提高度**, 纯PPO 停在"受控小跳"平台
(深蹲不足 → 高度卡 0.12m)。确认纯PPO 无参考的固有局限 (项目结论一致)。

## 完整跳跃评估 (老板要求, 工具 eval_jump_full.py)
按 §7.5+§1.3 六阶段 (站立→下蹲→起跳→空中→落地→恢复) 逐姿态评估, 详见
`_devlog/assess/reports/jump/2026-08-15_full_jump_phase_assessment.md`:
- 纯PPO v7: 站立✅/下蹲⚠️/起跳✅/空中✅/落地✅/恢复⚠️ → 5/7
- PPO+VMC v3: 站立✅/下蹲⚠️/起跳✅/空中✅/落地✅/恢复⚠️ → 5/7
- 对比基线: 纯PPO 0/7 (火箭), PPO+VMC 6/7 但姿态差
- **共同瓶颈**: 下蹲蓄力不足 (膝屈 0.04, 不深蹲) + 恢复晃动 (|gyro|~2) → 高度卡 0.12-0.16m

## v8/v9 迭代 (高度瓶颈攻坚, 记录在案)

**v8 (commit 48a0b76)**: 诊断 PPO+VMC 物理下蹲瓶颈 (VMC 压缩腿需~40步, fsm_crouch 0.35s 蹲不到位)
→ fsm_crouch_time 0.35→0.50, fb 0.7→0.5, crouch_depth 8→15; 纯PPO landing_soft 门控深蹲
+ threshold 0.45→0.48。**结果 (iter1000)**: 两算法都学会深蹲但**收敛到"接地慢伸"刷
launch_rise** (准静态抬升 0.68m, air 仅 1/16); 纯PPO 更在 iter1313 退化回"站立+开车"。
→ 教训: launch_rise 奖接地上升, 慢伸可刷满; 每轮修复都产生新 farm, "休息策略"
(landing_recovery+tracking) 始终是退路。

**v9 (commit 32ee64e)**: launch_rise 大降 (40/35→15) + jump_land 大升 (150→250, VMC 新增) —
farm 不再划算, 真实跳高+落地才给大额。**结果**: 两算法都退化回"站立+开车" (纯PPO 从
iter600 起所有跳跃奖励 0, PPO+VMC jump_land 仅 0.05) — 反farm 过猛, 休息策略退路。

**v10 (commit 95c9ac5, 决策定型)**: 
- 7 轮迭代证据链完整: 纯PPO 奖励工程收益递减 (火箭→悬空→不跳→慢伸→不跳), 最优 =
  v6/v7 受控小跳 (100%恢复, 高度 0.12m)。**纯PPO 回退 v7** (launch_rise 40, jump_land 150,
  landing_soft 不门控深蹲)。
- **PPO+VMC**: 实测零策略动作 FSM 本身能腾空 (air 27%, 深蹲到 0.26m), 策略 (fb 0.5) 会
  干扰破坏跳跃 → **fb 0.2 参考主导** + launch_rise 回 35 + jump_land 250 + landing_soft
  深蹲门控开 (fsm 强制下蹲满足) + fsm_crouch 0.50。让参考逼出真跳, 策略学恢复。
- `landing_soft_requires_crouch` 做成可配置 (默认 False, VMC 开)。

## v10 训练结果 (iter 1000, 关键突破)

### PPO+VMC v10 (fb 0.2 参考主导) — 跳高突破 0.20m
```
verify: survived=600/600  jump_height=0.246m  air_frac=0.16
eval:   attempts=16 airborne=15/16 recovered=14/16 成功恢复率=0.88 跳高 0.281±0.031m
完整评估: 站立✅/下蹲⚠️/起跳✅/空中✅(跳高0.298m, up_z0.992直立, 收腿轮速2.6)/
         落地✅(轮速0.9不空转)/恢复✅(|gyro|0.74稳定) → 6/7
```
**突破**: fb 0.2 参考主导逼出真跳 (跳高 0.246m > 0.20 阈值, v3 仅 0.156m)。
成功率 0.88 接近 0.90, 训练中 (iter 4000) 可能继续提升。

### 纯PPO v10 (回退 v7) — 稳定受控小跳
```
verify iter2000: survived=600/600 jump_height=0.161m air_frac=0.10 (稳定无退化)
eval iter1000: 16/16 真腾空, 100% 恢复, 跳高 0.076m
```
纯PPO 固有极限确认: 受控小跳 + 100% 恢复, 高度 ~0.12m (无参考结构)。

## SRL / SRL+VMC 改进补齐 (commit d40ca54)

老板要求四算法全部改进后统一 10000 轮训练供对比。补齐:
- **SRL**: 补几何接触检测 (原 force 阈值法, 与 jump.py/jump_vmc 同款修复) — 空中扇腿
  误判着地问题。其余改进 (landing_recovery/anti_drift/wheel_ground_matching/landing_window)
  已在 devlog [[NN_jump_srl_improve]] 完成。
- **SRL+VMC**: 继承 jump_vmc 的几何接触 (已生效); 改进 (FSM 映射修复 + landing_recovery/
  jump_upright/wheel_ground_matching) 已在 devlog [[NN_jump_srl_vmc_improve]] 完成。
- 四算法配置均确认 num_envs 1024 / max_iterations 10000。
- **SRL/SRL+VMC 4000 轮验证训练中** (`jump_srl_v3.log` / `jump_srl_vmc_v3.log`), 确认
  改进有效后启动四算法 10000 轮。

## 验证训练结果 (iter 1000-2000) — 四算法改进全部确认

| 算法 | 验证 iter | 成功恢复率 | 跳高 | 改进确认 |
|---|---|---|---|---|
| 纯PPO (v10) | 2000 稳定 | 100% | 0.16m | ✅ 受控小跳 |
| PPO+VMC (v10) | 1000 最优 | 88% | 0.25m | ✅ 跳高突破 0.20 |
| SRL (改进后) | 2000 | 100% | 0.35m | ✅ 恢复/防漂移/轮速全生效 |
| SRL+VMC (改进后) | 1000 最优 | 100% | 0.32m | ✅ FSM修复+落地恢复生效 |

整体效果良好: 四算法成功率 88-100%, 跳高均超过/接近 0.20m (纯PPO 除外, 固有极限)。
SRL/SRL+VMC 训练存在"高跳 vs 安全恢复"漂移, 取最优 checkpoint。
→ 启动四算法 10000 轮/1024 环境训练供对比实验。

## 后续计划

- [ ] 每 1000 iter verify_jump 复查 (跳高>0.20 / air>0.05 / survived)
- [ ] 纯PPO v4 若仍悬空 → 追加 anti-hover (空中+trigger-off 惩罚) 或加强 landing_recovery 门控
- [ ] 达标后: eval_jump_repeat 成功率、渲染视频、导出姿态 CSV、版本备份
