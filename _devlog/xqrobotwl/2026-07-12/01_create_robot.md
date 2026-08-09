# 01 基于 xqrobotV2 创建 xqrobotwl 机器人

**日期**: 2026-07-12
**来源**: 用户需求 — xqrobotwl 是 xqrobotV2 的简化碰撞体+优化质量版本
**关联**: `/home/robot/xiaoq/Myrobot/xqrobotwl/`

---

## 问题描述

xqrobotwl 是 xqrobotV2 的变体：简化碰撞几何（box/cylinder 替代 mesh）、调整质量/惯量、优化关节结构。需要为它创建完整的训练环境，与 xqrobotV2 保持功能对等。

## 解决方案

### 1. 资产层

| 文件 | 说明 |
|------|------|
| `assets/robots/xqrobotwl/xqrobotwl.xml` | MuJoCo XML（简化碰撞体） |
| `assets/robots/xqrobotwl/scene_flat.xml` | 平地场景 + keyframe (z=0.65) |
| `assets/robots/xqrobotwl/locomotion_task.xml` | 步态 keyframe |
| `assets/robots/xqrobotwl/meshes/` | 9 个 STL 网格 |

### 2. 环境层

从 xqrobotV2 完整复制并适配：

```
envs/locomotion/xqrobotwl/
├── base.py         # 关节/传感器常量
├── joystick.py     # 平地行走 (含 sign_flip)
├── rough.py        # 粗糙地形
├── stairs.py       # 楼梯专项 (NP3O)
├── jump.py         # 跳跃
├── toe_walk.py     # 脚趾行走
└── __init__.py
```

### 3. 与 xqrobotV2 的关键差异

| 维度 | xqrobotV2 | xqrobotwl | 处理方式 |
|------|------|------|------|
| 关节名 | `left_joint_1/2/3` | `joint_left_hip_roll/pitch/knee` | base.py 常量 |
| 右腿关节轴 | 全 +Y | 右 hip_pitch/knee/ wheel 为 -Y | `apply_action` sign_flip `[1,1,-1,1,-1,1,1,-1]` |
| 碰撞体 | mesh | box/cylinder | XML 直接定义 |
| 轮碰撞 | mesh 自动对齐 | cylinder 需 euler 旋转 | `euler="1.5708 0 0"` |
| 质量 | ~6.7kg | ~10.5kg | XML 惯性参数 |
| 默认角度 | `[-0.1,0.1,-0.1, 0.1,0.1,-0.1]` | `[0.1,0.15,0.15, -0.1,0.15,0.15]` | + sign flip 保持对称 |

### 4. 注册

`locomotion/__init__.py` 添加 `"unilab.envs.locomotion.xqrobotwl"`。

### 5. 配置和脚本

```
conf/ppo/task/xqrobotwl_{walk_flat,w walk_rough,jump_flat}/mujoco.yaml
conf/np3o/task/xqrobotwl_stairs/mujoco.yaml
shell/xqrobotwl/train_ppo_{flat,rough,jump_flat}.sh
shell/xqrobotwl/train_np3o_stairs.sh
shell/xqrobotwl/eval_*.sh
```

---

## 修改文件

| 文件 | 改动 |
|------|------|
| `src/unilab/assets/robots/xqrobotwl/` | 新建完整资产目录 |
| `src/unilab/envs/locomotion/xqrobotwl/` | 新建 7 个 py 文件 |
| `src/unilab/envs/locomotion/__init__.py` | 添加 xqrobotwl |
| `conf/ppo/task/xqrobotwl_*/` | 新建 3 个 YAML |
| `conf/np3o/task/xqrobotwl_stairs/` | 新建 1 个 YAML |
| `shell/xqrobotwl/` | 新建 8 个脚本 |
| `shell/xqrobotV2/` | 原脚本移入子目录 |

---

## 验证

```bash
bash shell/xqrobotwl/train_ppo_flat.sh
bash shell/xqrobotwl/train_ppo_rough.sh
bash shell/xqrobotwl/train_np3o_stairs.sh
bash shell/xqrobotwl/train_ppo_jump_flat.sh
```

---

*记录人: AI (opencode)*
