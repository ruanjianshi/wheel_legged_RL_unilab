# 04 全量核对: 13 个 xqrobotwl 任务机器人模型已对称化

## 日期
2026-08-14

## 来源
用户要求: 检查"8 个任务"的机器人模型是否更新成对称模型 → 用户进一步指正为 **13 个任务**。
对全部 13 个 xqrobotwl 任务逐一构建 env 验证模型状态。

## 修改了什么
无代码修改 — 纯**验证核对**。核对对象:
- 几何对称化 (devlog 03): `xqrobotwl.xml` + `xqrobotwl_vmc.xml` 左右轮镜像对称
- 碰撞对齐 (devlog 02): 髋碰撞删除 + 腿盒补旋转 (10→8 碰撞体)

## 验证方法
对 13 个任务配置逐个 `registry.make` 构建 env, 检查编译后模型:
1. 碰撞体数量 == 8 (URDF 对齐)
2. 无 `link_1_collision` 髋碰撞体
3. 左右轮 body 位置镜像对称 (|y| 差 < 2mm 且 z 差 < 2mm)

## 评估效果 (13/13 全部通过 ✅)

| # | 任务 | env | 碰撞数 | 髋碰撞 | 轮对称 |
|---|---|---|---|---|---|
| 1 | walk_flat | XqRobotWLWalkFlat | 8 ✅ | 无 ✅ | ✅ |
| 2 | toe_walk_flat | XqRobotWLToeWalkFlat | 8 ✅ | 无 ✅ | ✅ |
| 3 | walk_rough | XqRobotWLWalkRough | 8 ✅ | 无 ✅ | ✅ |
| 4 | jump_flat | XqRobotWLJumpFlat | 8 ✅ | 无 ✅ | ✅ |
| 5 | jump_srl_flat | XqRobotWLJumpSRLFlat | 8 ✅ | 无 ✅ | ✅ |
| 6 | jump_vmc_flat | XqRobotWLJumpVMC | 8 ✅ | 无 ✅ | ✅ |
| 7 | jump_srl_vmc_flat | XqRobotWLJumpSRLVMC | 8 ✅ | 无 ✅ | ✅ |
| 8 | backflip_flat | XqRobotWLBackflipFlat | 8 ✅ | 无 ✅ | ✅ |
| 9 | single_leg_flat | XqRobotWLSingleLegFlat | 8 ✅ | 无 ✅ | ✅ |
| 10 | single_leg_move | XqRobotWLSingleLegMove | 8 ✅ | 无 ✅ | ✅ |
| 11 | single_leg_unicycle | XqRobotWLSingleLegUnicycle | 8 ✅ | 无 ✅ | ✅ |
| 12 | fall_recovery_flat | XqRobotWLFallRecoveryFlat | 8 ✅ | 无 ✅ | ✅ |
| 13 | stairs | XqRobotWLStairs | 8 ✅ | 无 ✅ | ✅ |

**结论**: 13/13 任务全部解析到已对称的 `xqrobotwl.xml` 或 `xqrobotwl_vmc.xml`。
继承链覆盖: toe_walk/jump/single_leg* → WalkFlatEnv; backflip/fall_recovery/jump_srl* → JumpSRL→WalkFlatEnv; rough/stairs 直接引用。

## 说明
- 仓库另有 5 个 `xqrobotV2_*` 配置 (xqrobotV2 旧机器人): 轮子本来就对称, 碰撞用 mesh 方案, 不属于 xqrobotwl 任务, 未改动。
- 几何对称化会改变仿真物理 → RL 模型 (在不对称几何训练) 需在对称几何下重新评估/重训 (遗留)。

## 后续计划
- RL 模型对称几何下评估 (walk/rough/stairs 等)
- P4 rough 地形稳定性 (经典控制遗留)

## 关联日志
- [[03_wheel_symmetry_fix]] 几何对称化
- [[02_collision_align_urdf]] 碰撞对齐
- [[08_classic_symmetric_constants]] 经典控制常量对称化
