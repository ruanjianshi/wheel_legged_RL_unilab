# 10 — 站姿参数修复 + 高度目标修正 + 平台适配

## 日期
2026-07-22（续 09）

## 来源
训练监控发现 ep_len 停滞不增，深入诊断站姿与跳跃压制问题。

## 问题描述

### 1. ep_len 停滞（30 步，不增长）
- iter 148: ep_len=30.7，奖励几乎不增
- 根因：自适应 `feedback_gain=0.4`（站立阶段）不足够，DEFAULT_LEG_ANGLES 非稳定平衡需要更强控制

### 2. 跳越被压制（max_z=0.66m vs 旧模型 1.41m）
- `base_height=-60` 对称惩罚：跳越高越罚
- z=1.4m 时：跳跃奖励+24 vs 高度罚-33.8 = 净负
- 策略学到"不跳"以获取更高总分

### 3. 站姿不对称
- L_knee=-0.58（伸直）vs R_knee=+0.23（微弯）→ 左腿单独撑地
- L_hip_roll=+0.38（4×默认）vs R_hip_roll=-0.01 → 严重偏斜
- leg_mirror=-2 太弱，对标 PPO 的 -12

### 4. base_height_target 物理不可达
- 所有 xqrobotwl 任务使用 0.65m 目标
- 但实际 PPO 仅站到 0.52m，SRL 0.57m
- 0.65m 是 xqrobotV2 的配置，xqrobotwl 上限约 0.55-0.60m

### 5. 策略输出巨人值
- R_knee action max=+12.6，目标 >1.4 rad 超过关节限位 0.87
- PD 控制器饱和，策略在死区白费力气
- clip_actions=100 允许无界输出

## 解决方案

### 1. 自适应增益调整 (`jump_srl.py:461`)
```
站立(gain 0.4→0.8):  0.8×0.7=0.56 rad, 对标 PPO 0.6
```
### 2. 跳越阶段 base_height 归零 (`jump_srl.py:431-436`)
```
FSM 1/2/3(蹬/飞/落) 不罚高度, 仅靠 jump_height 奖励驱动
站立 -1/恢复 4 保持 -60 强罚
```

### 3. 对称约束 + 动作限制 (`mujoco.yaml`)
| 参数 | 旧 | 新 |
|------|----|----|
| `leg_mirror` | -2 | **-12** |
| `clip_actions` | 100 | **2.0** |

### 4. base_height_target 全局修正
所有 xqrobotwl 任务 `0.65→0.55`：
```
conf/ppo/task/xqrobotwl_walk_flat/mujoco.yaml
conf/ppo/task/xqrobotwl_walk_rough/mujoco.yaml
conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml
conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml
conf/np3o/task/xqrobotwl_stairs/mujoco.yaml
src/unilab/envs/locomotion/xqrobotwl/jump.py
src/unilab/envs/locomotion/xqrobotwl/jump_srl.py
```

### 5. 新建 toe_walk 配置
```
conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml (新建)
shell/xqrobotwl/train_ppo_toe_walk.sh (新建)
shell/xqrobotwl/eval_ppo_toe_walk.sh (新建)
```

### 6. 平台自动检测（全部 12 个脚本）
```bash
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="uv run mjpython"   # macOS
else
    PYTHON="uv run"            # Linux
fi
```

## 修改文件

| 文件 | 改动 |
|------|------|
| `jump_srl.py:461` | 站立 gain 0.4→0.8 |
| `jump_srl.py:431-436` | 跳越阶段 base_height 归零 |
| `conf/.../mujoco.yaml` | leg_mirror→-12, clip_actions→2.0 |
| `conf/.../mujoco.yaml` ×5 | base_height_target 0.65→0.55 |
| `jump.py`, `jump_srl.py` | default 0.65→0.55 |
| `conf/.../xqrobotwl_toe_walk_flat/` | 新建配置 |
| `shell/xqrobotwl/` ×2 | 新建 toe_walk 脚本 |
| `shell/xqrobotwl/` ×12 | 平台自动检测 |

## 验证方法

1. 最新 SRL 训练（2026-07-22_14-02-17）评估：
   - stand z=0.57m（vs 旧 0.47m，+21%）
   - jump max=0.66m（被 base_height 压制，待重训验证）
   - action_std=1.9（受控，不爆炸）
   - FSM 六态完整
2. 站姿关节详细监控：确认对称性问题需重训修复

## 训练状态
最新模型 iter 9999 完成，但跳跃被 base_height 压制。修复后需重训。

## 后续计划
1. 重训 SRL（完整修复后）
2. 监控 ep_len 在预热期增长（目标 iter 500+ 达到 5s 存活）
3. 跳跃期验证 max_z > 1.0m
4. 全部通过后运行消融实验

## 关联日志
- `2026-07-22/08_wheeled_srl_framework_diag.md` — SRL 框架初始构建
- `2026-07-22/09_fsm_warmup_adaptive_gain.md` — FSM + 预热修复
