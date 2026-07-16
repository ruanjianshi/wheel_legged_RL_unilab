# 04 键盘控制修复

**日期**: 2026-07-16
**来源**: 用户按键控制机器人不跳/不受控
**关联**: [03_posture_airtime](2026-07-16/03_posture_airtime.md)

---

## 问题描述

1. **按键 J 冲突**：J 是 MuJoCo 内置键（显示关节坐标），按 J 不跳
2. **自动在跳**：键盘模式下环境随机采样 jump_trigger，不按键也跳
3. **自动在走**：环境每 4 秒重采样 vx/vy/vyaw，不按方向键也走动

## 根因分析

| 问题 | 根因 |
|------|------|
| J 冲突 | MuJoCo viewer 硬编码 J=显示关节坐标，`_on_key` 未返回真值拦截 |
| 自动跳 | `_update_commands` 随机采样 jump_trigger∈[0,1]，未在键盘模式覆盖 |
| 自动走 | 同上，vx/vy/vyaw 随机采样 |

## 解决方案

### 1. J → H（避免 MuJoCo 冲突）

```python
# play_interactive.py
elif keycode == ord("H"):  # 原 ord("J")
    commander.jump_trigger = ...
```

### 2. `_on_key` 返回 1 拦截 MuJoCo

```python
def _on_key(keycode: int) -> int | None:
    ...
    return 1  # 消费按键，MuJoCo 不再处理
```

### 3. 键盘模式锁定全部指令

```python
# 每步写指令前先全部归零
env.state.info["commands"][:, :] = 0.0
# 只填键盘值
env.state.info["commands"][:, :3] = commander.command
# jump_trigger = 1 (按 H) 或 0
```

### 4. 键盘模式前已修项

| 修复 | 文件 |
|------|------|
| jump 课程绕过 | `play_interactive.py` `_init_commander` |
| jump 任务排除 height_target | 两处条件添加 `xqrobotwl_jump` |

## 修改文件

| 文件 | 改动 |
|------|------|
| `scripts/play/play_interactive.py:1016` | J→H |
| `scripts/play/play_interactive.py:1032` | legend J→H |
| `scripts/play/play_interactive.py:1219-1241` | `_on_key` 全部返回 1 |
| `scripts/play/play_interactive.py:1280-1282` | 每步 commands 全部归零 |

---

*记录人: AI (opencode)*
