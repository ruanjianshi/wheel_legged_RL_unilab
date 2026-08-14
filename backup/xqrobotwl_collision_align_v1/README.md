# xqrobotwl 碰撞体对齐 URDF — 版本备份 (v1)

日期: 2026-08-14
变更: 碰撞体对齐原始 URDF 源模型

## 备份内容
- `xqrobotwl.xml.orig` — **改动前**的 xqrobotwl.xml (含髋碰撞体 + 无腿盒旋转)
- `scene_flat.xml` — 改动前的平地场景

## 回滚方法
```bash
cp backup/xqrobotwl_collision_align_v1/xqrobotwl.xml.orig src/unilab/assets/robots/xqrobotwl/xqrobotwl.xml
# 如需一并回滚 VMC XML, 从 git 恢复 (xqrobotwl_vmc.xml 未单独备份, 见 git diff)
```

## 改动摘要 (v1 → v2)
- 删除左右髋碰撞体 (left/right_link_1_collision)
- 大腿/小腿碰撞体补 euler (2.48 / 0.7 rad)
- joystick.py `_LEG_GEOM_NAMES` 同步去掉髋名字

详见 `_devlog/xqrobotwl/common/2026-08-14/02_collision_align_urdf.md`

## 追加 (同日): 左右轮对称化
- 改动: left_link_wheel body pos (0.224,-0.037,-0.199) → (0.224,-0.031,-0.204) 镜像右轮
- 备份: xqrobotwl.xml.after_collision_align (碰撞对齐后、轮对称化前的版本)
- 效果: 轮世界坐标对称 (|y|=0.171), 轮距 0.342, 中点 y=0
- 显示站姿改镜像对称 (tools/mujoco/show_collision_model.py)
- 控制常量 (STANDING_ANGLES/LEG_TARGETS_COMPENSATED) 未动; P1 平衡 15s 仍存活
- 注意: 几何对称化影响所有用 xqrobotwl.xml 的 RL 环境 (walk/rough/stairs),
  RL 模型在不对称几何下训练, 可能需要重新评估
