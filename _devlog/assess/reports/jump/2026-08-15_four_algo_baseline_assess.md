# 四算法跳跃基线评估 + 问题根因分析

**日期**: 2026-08-15
**评估对象**: 四算法 08-14 对称几何重训基线 (每算法 model_9999)
**评估工具**: `verify_jump.py` / `eval_jump_repeat.py` / `diag_jump_trajectory.py`
**对应目标**: CLAUDE.md §7.5 平地跳跃 (成功率≥90% / 阶段链正确 / 轮速匹配 / 稳定落地恢复)
**状态**: 评估完成, 四算法基线效果与根因已定位

---

## 1. 结论摘要 (给老板)

| 算法 | 跳高 | 真实腾空 | 存活/成功率 | 达标? | 一句话结论 |
|---|---|---|---|---|---|
| **纯PPO** `jump_flat` | ~2.9m(失控) | air 97% 但不落地 | **0%** (16/16 摔) | ❌ 崩溃 | 火箭发射, 完全失控, 不是跳跃 |
| **SRL** `jump_srl` | 0.59m ✅ | air 17% ✅ | 100% ✅ | ⚠️ 两项硬伤 | 四算法最好, 但空中轮空转3.7m/s + 落地漂移2.7m |
| **PPO+VMC** `jump_vmc` | 0.28m ⚠️ | air 2% ❌ | 100% | ❌ 假跳 | 能站能恢复, 但几乎不腾空 (准静态伸腿) |
| **SRL+VMC** `jump_srl_vmc` | 0.42m ✅ | air 15% ✅ | **81%** ❌ | ❌ 落地不稳 | 真实跳跃, 但落地塌陷/俯仰倾倒, 成功率<90% |

**一句话**: 只有 SRL 做到了"能跳且能落地", 但轮速匹配和微动平衡不过关;
其余三算法各有致命问题 (失控火箭 / 假跳高 / 落地失败), **全部未达 §7.5**。

---

## 2. 评估方法

- `verify_jump.py`: 600 步, 触发脉冲 (200 步周期 100/100), 确定性策略 (action mean)
- `eval_jump_repeat.py`: 8 跳 × 2 episode = 16 次尝试, 每跳 settle50 + on100 + off120, 统计成功恢复率
- `diag_jump_trajectory.py`: 逐步输出 base_z / 轮地接触 / FSM 状态 / phase
- 环境按训练 `run_config.json` 重建 (VMC/feedback_gain 等与训练一致)

---

## 3. 详细评估数据

### 3.1 verify_jump (600 步基线)

| 算法 | survived | max_base_z | standing_z | jump_height | air_frac |
|---|---|---|---|---|---|
| 纯PPO | 106/600 | 2.962 | 0.000 | 2.962 | 0.97 |
| SRL | 300/300* | 1.022 | 0.433 | 0.589 | 0.17 |
| PPO+VMC | 600/600 | 0.775 | 0.480 | 0.295 | 0.02 |
| SRL+VMC | 600/600 | 0.945 | 0.476 | 0.470 | 0.15 |

*SRL 本次烟测 300 步; devlog 记录 600/600, 跳高 0.661, air 0.19。

### 3.2 eval_jump_repeat (16 次尝试 / 算法)

| 算法 | airborne | recovered | terminated | 成功恢复率 | 跳高(m) | 恢复窗口漂移(m) | 空中轮速峰值 |
|---|---|---|---|---|---|---|---|
| 纯PPO | 0/16 | 0/16 | 16/16 | **0.00** | — | — | — |
| SRL | 16/16 | 16/16 | 0/16 | **1.00** | 0.591±0.065 | 0.574±0.686 (max 2.67) | **57.3 rad/s (3.7m/s)** |
| PPO+VMC | 16/16 | 16/16 | 0/16 | **1.00** | 0.281±0.020 | 0.182±0.156 | 10.4 rad/s |
| SRL+VMC | 14/16 | 13/16 | 3/16 | **0.81** | 0.424±0.034 | 0.362±0.362 | 9.5 rad/s |

---

## 4. 各算法问题与根因

### 4.1 纯PPO (`jump.py`) — 崩溃: "不跳"是假象, 实际是火箭发射

**现象**: 老 verify 报 jump_height=0.004 (不跳), 但用几何接触检测修正后:
**base_z 冲到 2.9m、97% 步数腾空、第 106 步摔倒终止、16/16 全摔**。

**根因链** (diag + 训练 reward log 证实):
1. **轮地接触检测 bug (核心)**: `_update_wheel_contact` 用 `norm(wheel_force)>10` 判着地。
   空中扇腿时轮被腿加速, 轮 site 力 >10N (轮自重 ~23N) → 腾空被误判为着地 → 以
   `air_factor`/`wheel_contact` 门控的 `jump_height`/`wheel_air_time`/`landing_soft`
   **全部恒为 0**。
2. **奖励被 hack**: 策略永远因腾空得分 → 只拿 `launch_rise` (scale 40, 只看 base_z 不看
   接触) → 最优解 = "起飞越高越好, 不管落地" = 火箭。训练末期 reward: launch_rise≈31.4
   主导, jump_height≈0.002, wheel_air_time≈-0.03。
3. **无参考轨迹**: 纯 PPO 无 SLIP-FSM 参考, 稀疏奖励下找不到"受控跳跃"解, 落入高飞盆地。

**已修**: 08-15 已定位并修复 (几何接触检测 + 奖励重平衡), 但重训被中断, 需重新训练验证。

### 4.2 PPO+VMC (`jump_vmc.py`) — 假跳高: 能站能恢复, 但几乎不腾空

**现象**: 存活 100% / 恢复 100%, 但跳高仅 0.28m、air_frac 0.02、每跳 air_steps 仅 1-6 步。
diag: FSM 下蹲到 0.42 后, base_z 只升到 0.75, 轮地接触全程 [1,1] —— **准静态伸腿, 轮不离地**。

**根因链**:
1. **full-action 参考叠加 (fb=1.0)**: `final_L0 = 参考 + 1.0×策略动作`, 策略保留全部权限,
   能完全盖过 SLIP-FSM 蹲→蹬参考。
2. **策略学会慢伸白拿**: 下蹲相慢慢伸腿 (L0 0.29→0.56), 物理上准静态抬升轮不离地, 却刷满
   `launch_rise` (scale 40)。训练 reward: launch_rise=4.52 主导, jump_height=0.007。
3. **jump_height 被 air_factor 门控 → 永不激活** → 策略对真实跳跃零回报, 困在假跳高解。

**已修**: 08-15 已改 fb=0.7 (参考主导) + launch_rise 门控 FSM 蹬伸相 + anti_early_extend,
重训被中断。

### 4.3 SRL+VMC (`jump_srl_vmc.py`) — 真实跳跃但落地不稳, 成功率 81% < 90%

**现象**: 真实腾空 (air_steps 19-32), 跳高 0.42m, 但 16 次中 3 次终止 (81%)。
失败模式: 蹬伸期俯仰过大倾倒 / 落地硬冲击塌陷 (base_z 0.70→0.28) / 落地后缓慢塌缩摔倒。
diag: 落地后 base_z 塌到 0.275, 恢复慢且晃动 (0.44→0.60→0.45)。

**根因链**:
1. **FSM 阶段映射错位 (核心)**: SLIP-FSM 六态 (-1 idle,0 crouch,1 thrust,2 flight,
   3 prelanding,4 landing_absorb), 但 base `_jump_leg_reference` 把 state3 当"落地压缩"
   (0.42→0.30), state4 用默认增益 → **实际触地冲击发生在 state4 却用普通增益, 冲击不吸收**。
2. **无 landing_recovery 奖励**: 落地后无强激励恢复直立 → 学不到"跳完站稳"。
3. **landing_soft 门控错**: 绑 phase>=35 (触发窗口内), 落地时 trigger 已关 → 落地缓冲≈0。
4. **蹬伸俯仰无抑制**: orientation -5 太弱, vertical_thrust 主导, 策略为跳高容忍俯仰。
5. **wheel_ground_matching 死奖励**: config 有 scale 但 SRLVMC 无该 reward fn → 从未激活。

**已修**: 08-15 已修 FSM 映射 (state3 保持伸腿准备触地, state4 落地吸收 + 高阻尼) +
landing_recovery / jump_upright / wheel_ground_matching, 重训被中断。

### 4.4 SRL (`jump_srl.py`) — 最好, 但轮速匹配 + 微动平衡两项硬伤

**现象**: 跳 0.59m / air 17% / 存活 100% / 成功率 100% —— 四算法中唯一"能跳且能落地"。
但:
- **空中轮速空转 57 rad/s ≈ 3.7 m/s**: 违反 §7.5 轮速匹配。落地点地必然打滑。
- **恢复窗口漂移 0.574±0.686 m (单跳最大 2.67 m)**: 违反微动平衡。
- 落地塌陷到 0.27, 恢复慢且晃动 (diag: 落地 0.68→0.39→恢复 0.63 后仍晃动)。

**根因**:
1. **wheel_ground_matching 奖励从未生效 (bug)**: config scales 键是 `wheel_ground_matching`,
   但 `jump_srl.py` `_reward_fns` 注册键是 `wheel_air_time` → 键不匹配静默跳过 → 轮速匹配
   完全未训练。空中无阻力 + 无惩罚 → 轮自由转。
2. **action_magnitude 奖励从未生效 (bug)**: 同上键无 fn。
3. **无 anti_drift**: 落地塌陷反弹 + 无轮速匹配 → 前冲动量未吸收 → 站立期持续滑动。

**已修**: 08-15 已修 fn 对齐 + landing_recovery / anti_drift / landing_window 门控,
resume 重训被中断。

---

## 5. 共性根因 (跨算法)

1. **稀疏奖励 + 稀疏真实腾空信号**:
   所有算法的跳跃奖励都依赖 `wheel_contact` 判腾空; 接触检测在"空中扇腿"下失真 (力阈值法),
   导致腾空信号不可靠 → 纯 PPO 完全拿不到腾空奖励 (火箭), VMC 变体被假跳高解困住。

2. **SLIP-FSM 参考是控制力的关键**:
   - 有参考 + 策略受限 (SRL) → 真实跳跃
   - 无参考 (纯 PPO) → 完全失控
   - 参考被策略覆盖 (PPO+VMC full-action) → 假跳高
   - 参考正确 + 策略残差 (SRL+VMC residual) → 真实跳跃但落地环节弱

3. **奖励工程 bug 多 (静默失效)**:
   `wheel_ground_matching`/`action_magnitude` 键与 fn 不匹配、`landing_recovery` upright
   符号 bug、`landing_soft` 门控错位 —— 多个奖励"设计了但从未生效", 训练结果与设计意图不符。

4. **落地恢复是共同瓶颈**:
   除 SRL 外全部卡在"落地后恢复站立"; §7.5 的 ≥90% 成功率本质上要求"能跳 + 落地能恢复",
   落地缓冲 (腿压缩吸收动能) + 落地恢复 (双轮着地直立) 是短板。

---

## 6. 建议下一步

1. **恢复重训** (今天 08-15 四算法改进版训练已中断, run 目录被删):
   - 纯PPO: 几何接触修复 + 奖励重平衡 (jump_v2) — 需确认能从火箭盆地跳出
   - PPO+VMC: fb=0.7 + 蹬伸相门控 launch_rise + anti_early_extend
   - SRL+VMC: FSM 映射修复 + landing_recovery/jump_upright
   - SRL: fn 对齐 + anti_drift (resume 微调)
2. **每 1000 iter 用 verify_jump 复查** (跳高/air_frac/survived), 防假跳高与火箭复发。
3. **达标后** 渲染视频 + 导出姿态 CSV + 版本备份 (§6.1)。
4. **修复渲染工具 bug**: `render_jump_repeat.py` 用 `env.reset().obs`, 当前 env API 返回
   tuple, 渲染评估视频需先修。

---

## 附: 本次运行命令 (可复现)

```bash
# 验证 (600 步)
uv run tools/xqrobotwl/verify_jump.py --task <TASK> \
  --checkpoint logs/rsl_rl_ppo/<TASK>/2026-08-14_<run>/model_9999.pt --steps 600

# 重复跳跃成功率 (16 次)
uv run tools/xqrobotwl/eval_jump_repeat.py --task <TASK> \
  --checkpoint .../model_9999.pt --jumps 8 --episodes 2

# 逐步轨迹诊断
uv run tools/xqrobotwl/diag_jump_trajectory.py --task <TASK> \
  --checkpoint .../model_9999.pt --steps 300
```
