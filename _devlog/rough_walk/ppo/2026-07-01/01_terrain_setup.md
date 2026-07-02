# 01 构建地形可视化工具 + 修复 MuJoCo 工具

**日期**: 2026-07-01  
**来源**: 不平坦地面 PPO 训练前需要预览地形

---

## 问题描述

1. 缺少地形预览工具，无法在训练前查看 XqRobotV2WalkRough 使用的地形
2. `show_base_frame.py` 在绘制坐标轴时有 NaN 除零警告
3. `show_terrain.py` 初次编写时 terrain_origins 格式化错误

## 修改文件

| 文件 | 改动 |
|------|------|
| `tools/mujoco/show_terrain.py` | **新建** — 生成 XqRobotV2WalkRough 同款地形 PNG，构建 MuJoCo 场景 |
| `tools/mujoco/show_base_frame.py:48-63` | 修复轴方向正交化（NaN 保护） |

### show_terrain.py 要点

- 使用 `TerrainGenerator` 生成 6×6 地形网格（8×8m）
- 子地形：40% 随机粗糙 + 40% 波浪 + 10% 上坡 + 10% 下坡
- `border_width` 必须为 `horizontal_scale=0.1` 的整数倍（已设置 0.2）
- MuJoCo 场景必须写入临时文件（需与 robot XML 同目录才能使用相对 meshdir）

### show_base_frame.py 修复

```python
# 原版: ref 选择可能产生与 d 平行的向量 → cross=0 → NaN
ref = np.array([0, 0, 1]) if abs(d[2]) > 0.9 else np.array([1, 0, 0])
y = np.cross(d, ref)
y /= np.linalg.norm(y)  # ← NaN

# 修复: 加保护，备用 ref
yn = np.linalg.norm(y)
if yn < 1e-10:
    ref = np.array([0, 1, 0])
    y = np.cross(d, ref)
    yn = np.linalg.norm(y)
y /= max(yn, 1e-10)
```

## 验证

```bash
# 地形生成通过（无报错）
uv run tools/mujoco/show_terrain.py
# 输出: Terrain OK. nbody=10

# 坐标系显示正常（无 NaN）
uv run tools/mujoco/show_base_frame.py
```

## 发现

`XqRobotRoughTerrainCfg` (rough.py) 中所有子地形的 `border_width` 已正确设为 0.2，与 `horizontal_scale=0.1` 兼容。无需修改。

## 后续计划

- [ ] 启动 XqRobotV2WalkRough PPO 训练
- [ ] 训练中定期使用 assess 框架评估

---

*记录人: AI (opencode)*
