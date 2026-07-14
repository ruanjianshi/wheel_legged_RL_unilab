# 03 丰富地形配置 + 视觉增强

**日期**: 2026-07-01  
**来源**: 训练前地形预览不够明显  
**关联**: [01 地形可视化工具](../../tools/2026-07-01/01_terrain_viewer_and_fixes.md)

---

## 问题描述

1. XqRobotV2 粗糙地形仅 4 种类型（随机/波浪/上下坡），缺少阶梯
2. 地形在 MuJoCo 中呈单色平面，无法分辨高度变化

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/envs/locomotion/xqrobotV2/rough.py:73-101` | 添加 `pyramid_stairs` + `pyramid_stairs_inv`，调整比例 |
| `tools/mujoco/show_terrain.py` | 重写，使用 rough.py 的配置 + 多光源 + 纹理材质 |

### rough.py 地形比例

| 改前 | 改后 | 类型 |
|------|------|------|
| — | 15% | 上楼梯 (step 0.02-0.08m) |
| — | 15% | 下楼梯 (inv) |
| 40% | 30% | 随机粗糙 (noise 0.01-0.06m) |
| 40% | 30% | 波浪 (amplitude 0-0.10m) |
| 10% | 5% | 上斜坡 (slope 0-0.15) |
| 10% | 5% | 下斜坡 (inv) |

> 参考 Go2 的 `Go2RoughTerrainCfg`（go2/rough.py:143-185）

### show_terrain.py 视觉改进

- 3 方向光源（主光 + 侧光×2），增强阴影
- 棋盘纹理 groundplane，放大到 10×10 重复
- 降低 reflectance (0.15) 减少反光
- 深色基调 (rgb 0.15-0.35)，白色网格线标记

## 验证

```bash
uv run tools/mujoco/show_terrain.py
```

## 后续计划

- [ ] 启动 XqRobotV2WalkRough PPO 训练

---

*记录人: AI (opencode)*
