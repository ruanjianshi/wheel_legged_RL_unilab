# [NN] 纯PPO jump_flat 根因修复: 轮地接触检测 bug → 火箭发射假象 (0.004m 实为 2.9m 不落地)

**日期**: 2026-08-15
**来源**: 任务启动基线复现: `verify_jump.py` 报 `jump_height=0.004 / air_frac=0.00 / survived=106/600`。要求向 §7.5 目标改进纯 PPO 平地跳跃。
**关联**: [[28_pure_ppo_reward_hacking_v10_v12]], [[27_landing_soft_exploit_fix]]

---

## 问题描述

复现基线 (08-14 全量 model_9999):

```
task=XqRobotWLJumpFlat survived=106/600 terminated=True max_base_z=3.109 standing_z=3.106 jump_height=0.003 air_frac=0.00
```

看似"完全不跳 (0.004m)" + "106 步后下蹲跌倒终止"。**实际是假象。**

## 根因分析 (诊断链)

1. **逐步诊断 (`diag_jump_trajectory`)**: 机器人触发窗口内 base_z 从 0.39 连续升到 **3.1m**, 而 `wheel_contact` 恒为 [1,1] (轮子"始终着地")。
2. **几何接触重算 (`left_wheel_world_pos` framepos 传感器)**: 用轮心世界 z < 0.13m (轮半径 0.11 + 余量) 判定着地, 同一策略实测:
   ```
   GEOMETRIC-air: survived=106/600 terminated=True max_base_z=2.922 standing_z=0.000 jump_height=2.922 air_frac=0.97
   ```
   → **旧模型其实是"火箭发射": 97% 步数腾空, 飞到 2.9m, 从不落地, 第 106 步摔倒终止**。不是不跳, 是跳得完全失控。
3. **contact bug 机理**: `_update_wheel_contact` 用 `norm(wheel_force) > 10` 判着地。腾空时若腿部还在动 (纯PPO 空中扇腿), 轮被腿加速, 轮 site 力 > 10N (轮自重 ~23N), 所以腾空被误判为着地。
   - 后果: 以 `air_factor`/`wheel_contact` 门控的 `jump_height` / `wheel_air_time` / `landing_soft` 全部永远为 0 → 策略**从不因腾空得分**, 只拿 `launch_rise` (仅用 base_z, 不看接触) → 最优解 = 起飞越高越好, 不管落地 → "火箭"。
   - SRL 能跳 0.605m 且 air 检测正常, 因为 SRL 参考 FSM 让空中收腿不动 → 腾空时轮力 ~0 → 接触正确变 0。纯PPO 无参考, 空中乱动 → contact 永远 1。
4. **训练期奖励佐证** (08-14 run events): 末期 `launch_rise≈31.4` (scale 40 主导), `jump_height≈0.002`, `wheel_air_time≈-0.03`, `joint_action_rate≈-2.98` (空中扇腿罚) → 策略在"空中高飞"盆地收敛。

## 解决方案

### 1. 轮地接触检测修复 (env, 按 task 隔离)

`src/unilab/envs/locomotion/xqrobotwl/jump.py`:
- 新增 `_update_wheel_contact_geom()`: 用 `left_wheel_world_pos` framepos 传感器读轮心世界 z, `z < 0.13` (轮半径 0.11 + 0.02 穿透余量) 判着地。对空中扇腿免疫 (轮心离地 > 0.13 即判腾空)。左轮镜像到右轮 (XML 只有 left 传感器)。
- `XqRobotWLJumpFlatEnv.update_state` 改用 `_update_wheel_contact_geom`。

**隔离性**: VMC / VMC+SRL 的 `update_state` 是独立实现 (直接调 force 版 `_update_wheel_contact`), SRL 是独立文件, 均不受影响。`pytest tests/envs/locomotion/xqrobotwl -k jump` 10 passed。

### 2. 奖励重平衡 (config, 只改 jump_flat)

`conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml`:

| 项 | 旧 | 新 | 理由 |
|----|----|----|----|
| `launch_rise` | 40 | 20 | 旧 40 驱动"火箭" (腾空后还按着地算 rise); 降后"受控跳"才能跑赢 |
| `jump_height` | 15 | 30 | 腾空最高 ~60/步, 让真跳收益压倒原地拉伸/火箭 |
| `wheel_air_time` | 20 | 30 | 同上 |
| `landing_soft` | 20 | 30 | 强化落地缓冲收益 (真实腾空落地后才激活) |
| `landing_recovery` | 4 | 10 | 强化落地后恢复站立 |
| `joint_action_rate` | -0.1 | -0.15 | 反空中扇腿乱动 |
| `wheel_action_rate` | -0.015 | -0.02 | 同上 |
| `entropy_coef` | 0.003 | 0.004 | 略增探索, 帮助跳出"火箭"盆地 (0.005 曾致落地乱) |

## 验证方法

1. 几何接触与 force 接触在旧模型上对照: 0.97 vs 0.00 air → 确认 contact 是假象根源。
2. `pytest -k jump` 10 passed; `ruff` clean。
3. **重训** (contact 修复后旧模型权重已失效, 必须重训): 先 4000 iter 验证, 每 1000 iter `verify_jump.py` 查跳高/存活。
4. 达标判定: 重复跳 ≥10 次, 成功率 ≥90%, 阶段姿态正确, 落地后恢复站立 (§7.5)。

## 训练后效果

(训练中, 见 `logs/train/jump_v2.log`; 评估数据待补)

## 参数调整好坏

(待评估补)

## 后续计划

- [ ] 监控 jump_v2 训练 (4000 iter), 每 1000 iter verify
- [ ] 若 launch_rise 仍被"原地拉伸/高飞"利用 → 将 rise 基准从 `window_min_z` 改为固定站高 0.55 (env 隔离改 `_reward_launch_rise`)
- [ ] 有效则全量 10000 iter
- [ ] 达标后: 姿态 CSV + 渲染视频 + 版本备份
