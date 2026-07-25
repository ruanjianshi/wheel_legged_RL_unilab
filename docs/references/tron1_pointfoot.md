# tron1 点足行走参考

## 来源
`/home/robot/xiaoq/projects/tron1-rl-isaaclab` — Isaac Lab + RSL-RL PPO

## 核心设计

**无参考轨迹**。步态完全从约束中涌现：

### GaitReward (相位门控 + 接触约束)

```
摆地:  phase ∈ [0.5, 1.0] → 罚足端接触力
支撑:  phase ∈ [0.0, 0.5] → 罚足端移动速度
```

- 不指定关节角度，只约束行为结果
- 惩罚 > 奖励: `tracking_contacts_shaped_force = -2.0`
- 使用 Von Mises 平滑过渡 (kappa=0.05)
- 频率 1.5-2.5 Hz, offset=0.5(交替), duration=0.5(50%占空比)

### feet_regulation (足端滑移约束)

```
reward = -0.1 * |v_horiz| * exp(-h / 0.65)
```

- 近地时 (h↓): exp≈1 → 强罚滑移 → 逼抬腿
- 离地时 (h↑): exp≈0 → 放松 → 自由摆动

### foot_landing_vel (软着陆)

```
即将着陆 = h < 0.08 AND 未着地 AND 向下运动 → 罚垂直速度
```

## 对我们的启发

我们的 `swing_lift` (reward, +20) 不起作用因为策略宁愿不拿分也不冒险抬腿。
应该改为 `swing_contact_penalty` (penalty, -20) — 摆地时只要轮子着地就罚。

加上 `feet_regulation` — 轮子近地时罚滑移, 离地时不罚 → 逼策略把轮子抬高。

## 关键参数

| 参数 | 值 |
|------|-----|
| base_height_target | 0.65m |
| 步频 | 1.5-2.5 Hz |
| 占空比 | 50% |
| 相位移 | 0.5 (交替) |
| action_scale | 0.25 |
| num_envs | 4096 |
