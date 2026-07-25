# 13 — 粗糙地形增强 + 高度命令课程

## 日期
2026-07-22

## 来源
用户要求不平坦地形更陡峭 + 腿长自适应 + 高度指令控制。

## 工作内容

### 1. 粗糙地形难度提升

| 地形 | 旧占比 | 新占比 | 旧范围 | 新范围 |
|------|--------|--------|--------|--------|
| flat | 20% | **5%** | — | — |
| random_rough | 35% | 30% | 0.005-0.04m | **0.02-0.08m** |
| wave | 35% | 25% | 0-0.12m | **0.03-0.18m** |
| slope | 5% | **20%** | 0.1-0.35 | **0.15-0.50** |
| slope_inv | 5% | **20%** | 0.1-0.35 | **0.15-0.50** |

### 2. 高度命令自适应

**修改 `joystick.py:560`**: `base_height` 奖励目标从固定值改为命令高度
```
旧: reward = (actual_h - 0.55)^2 × -5
新: reward = (actual_h - commands[4])^2 × -5
```
→ policy 学会根据高度命令动态调节腿长（伸腿站高 / 屈膝下蹲）

### 3. 高度课程

**修改 `joystick.py` curriculum config**: 新增 `height_step=0.001`, `min_height_frac=0.5`

**修改 `_update_curriculum`**: 高度维度随 vx/vyaw 一同展开
```
iter 0:     height ≈ 0.55 ± 0.03 (窄)
iter →:     逐步展开
iter 全开:  height ∈ [0.40, 0.90] (宽)
```

### 4. 可视化更新

`tools/mujoco/show_terrain.py`: 改用 xqrobotwl terrain config + xqrobotwl 机器人模型

## 修改文件

| 文件 | 改动 |
|------|------|
| `rough.py:72-100` | 地形比例和难度调整 |
| `rough.py:35-40` | 导入 pyramid_stairs (后移除) |
| `joystick.py:560` | base_height_target = commands[:, 4] |
| `joystick.py:71-80` | curriculum 新增 height_step |
| `joystick.py:614-628` | _update_curriculum 加高度扩展 |
| `tools/mujoco/show_terrain.py` | xqrobotwl 适配 |
| `shell/xqrobotwl/*.sh` ×12 | macOS/Linux 平台自动检测 |

## 验证
- 高度命令已存在于 5D 命令（第 5 维），policy 可观测
- rough env 从 joystick 继承奖励计算，自动获得动态高度目标
- 高度扫描 (187D) 提供地形预览给 critic
- 键盘 Q/E 可在 play 模式调节高度

## 关联日志
- `2026-07-22/10` — 高度目标统一修正
