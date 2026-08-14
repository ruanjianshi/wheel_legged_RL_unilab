# 03 左右轮对称化 + 显示站姿镜像对称 (修复"左腿歪")

## 日期
2026-08-14

## 来源
用户在看碰撞体查看器时反馈: **"左腿有点歪, 调整, 和右腿对称"**。

## 修改了什么

### 1. 轮几何对称化 (物理)
`xqrobotwl.xml` + `xqrobotwl_vmc.xml` 第 40 行:
`left_link_wheel` body pos `(0.224, -0.037, -0.199)` → `(0.224, -0.031, -0.204)` (镜像右轮)

### 2. 显示站姿对称化 (视觉)
`tools/mujoco/show_collision_model.py`:
STANDING_ANGLES 从 RL 测得的不对称值 `[0.102, 0.083, -0.079, 0.013, -0.108, 0.019]`
→ 镜像对称 `[0.0, 0.1083, -0.0188, 0.0, -0.1083, 0.0188]`

### 3. 渲染工具
`tools/mujoco/render_collision_model.py` 新增 (离屏绕机 360° 渲染碰撞体视频, MjvOption/MjvCamera API)

## 哪些文件
- `src/unilab/assets/robots/xqrobotwl/xqrobotwl.xml` (左轮 pos)
- `src/unilab/assets/robots/xqrobotwl/xqrobotwl_vmc.xml` (左轮 pos)
- `tools/mujoco/show_collision_model.py` (对称站姿)
- `tools/mujoco/render_collision_model.py` (新增渲染)
- 备份: `backup/xqrobotwl_collision_align_v1/` (碰撞对齐后、轮对称化前版本)

## 效果 (数值验证)
对称化后 (显示站姿下):
| 项 | 前 | 后 |
|---|---|---|
| 左轮世界 y | -0.219 | **-0.171** |
| 右轮世界 y | +0.166 | **+0.171** |
| 轮\|y\|差 | 53 mm | **0.000** ✅ |
| 轮距中点 y | -0.0265 (偏左) | **0.0000** (正中) ✅ |
| 大腿/小腿 y | 不对称 | ±0.123 / ±0.128 完全镜像 ✅ |

## 参数调整好坏
- **未动控制常量**: `STANDING_ANGLES`(控制重置)和 `LEG_TARGETS_COMPENSATED`(腿伺服目标)保持原值。
- 平衡回归: **LQR P1 15s 仍存活** (gyro 0.206 ✅, yaw 0.176°, 与改动前 0.096° 同量级) —
  轮几何对称化没有破坏平衡。
- 注意: 用对称站姿作为控制重置也能平衡 (yaw 7.6°, 仍 <30°), 但原不对称站姿 yaw 更小 (0.18°),
  故控制重置保持原值不动, 只修显示。

## 根因分析
"左腿歪" 有**两个来源**:
1. **几何**: 左右轮 body 位置不对称 (53 mm) — 已物理修复 (镜像右轮)
2. **显示站姿**: 原 RL 测得的自然站姿左右不对称 (L_hip_roll=0.102 外撇 vs R≈0) — 已用镜像对称站姿

## 验证方法
1. 对称性数值验证: 轮/腿世界坐标 |y| 差 = 0 ✅
2. LQR P1 平衡回归: 15s 存活 ✅
3. 渲染视频 `video/classic/collision_model_symmetric.mp4` (120 帧 360° 绕机)
4. 交互查看器重新打开, 用户目视确认

## ★ 影响与遗留 (如实)
- **RL 影响**: 轮几何对称化改变了 xqrobotwl.xml 物理, 影响 walk/rough/stairs 三个 RL 环境。
  RL 模型在**不对称几何**下训练, 现在跑在对称几何上 → 平衡/行走策略可能需要重新评估或重训。
- **真机对应**: 对称化后仿真与原始 CAD (不对称) 不再一致; 若真机左轮确实偏 53mm, 需以真机为准。
- 控制重置仍用不对称站姿 (yaw 更小), 但几何已对称 — 若用户希望运行中也完全对称, 需改控制基线 (另议)。

## 后续计划
- 请用户目视确认对称模型; 运行 RL 评估看对称几何下的影响
- 若决定改控制站姿为对称, 需重新验证 P2/P3 并更新 LEG_TARGETS_COMPENSATED

## 关联日志
- [[02_collision_align_urdf]] 碰撞体对齐
- [[01_collision_model_viewer]] 碰撞体显示 + 轮不对称发现
