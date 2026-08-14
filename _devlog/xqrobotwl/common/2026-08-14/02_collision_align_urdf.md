# 02 碰撞体对齐 URDF 源模型 (删髋碰撞 + 补腿盒旋转)

## 日期
2026-08-14

## 来源
用户要求: 在 `tools/mujoco/` 显示 xqrobotwl 碰撞体(01 工具), 对比发现 MuJoCo 碰撞体与
原始 ROS URDF (`/home/robot/xiaoq/Myrobot/xqrobotwl/urdf/xqrobotwl.urdf`) 不一致 → 用户决定"对齐,修改"。

## 修改了什么

### 对比结论 (URDF = 源, MuJoCo 需对齐)
| 连杆 | URDF | MuJoCo 原状 | 处理 |
|---|---|---|---|
| 髋 L/R | **无碰撞体** (注释: hip rarely contacts ground) | 多加了 box 0.04×0.06×0.04 | **删除** |
| 大腿 L/R | box 带 rpy(0, 2.48, 0) | 无旋转 (轴对齐) | **补 euler="0 2.48 0"** |
| 小腿 L/R | box 带 rpy(0, 0.7, 0) | 无旋转 | **补 euler="0 0.7 0"** |
| base 上下箱 / 轮 | 尺寸位置一致 | 一致 | 不动 |

### 改动文件
- `src/unilab/assets/robots/xqrobotwl/xqrobotwl.xml`(6 处: 删 2 髋碰撞 + 4 腿盒加 euler)
- `src/unilab/assets/robots/xqrobotwl/xqrobotwl_vmc.xml`(同 6 处, jump/backflip 用同一机器人)
- `src/unilab/envs/locomotion/xqrobotwl/joystick.py:185-192`: `_LEG_GEOM_NAMES` 删掉髋 2 个名字
  (原引用了已删除的 geom, `randomize_leg_length` 若开启会 KeyError)

### 备份 checkpoint
`backup/xqrobotwl_collision_align_v1/` — 原始 xqrobotwl.xml + scene_flat.xml

## 验证 (无辅助确定性评估)
- 两 XML 均编译成功, 碰撞体 10→**8** (与 URDF 一致), 髋碰撞消失, 腿盒带旋转 (世界 x 轴指向后下)
- **LQR P1 平衡 15s**: gyro 0.211 / linvel 0.024 / yaw 0.096° — **与改动前逐位一致** (轮是唯一触地面, 碰撞对齐零影响)
- `randomize_leg_length` 开启路径: 无 KeyError, geom 缩放生效
- jump_vmc / stairs env 构建步进正常
- 门禁: `make format` ✅, `make type` ✅ (0 err), pytest ✅ (796 passed)

## ★ 发现: P4 粗糙地形当前不稳定 (非本次改动导致)
- `balance_lqr.py --phase 4` 所有种子 (0/1/2/42) 均在 <0.5s 终止 (tiny tilt)
- **A/B 测试**: 用备份的原始 XML (改动前) 同种子跑 P4, 同样 0.09s 终止 → **碰撞对齐不是原因**
- 髋锁开/关测试: 均失败 → 非髋锁原因
- 控制器参数/配置在重构后逐项核对与旧代码一致 → 非重构原因
- 结论: rough 地形 spawn 附近有 ~9cm 起伏, 零命令期站立在起伏上本就苛刻; devlog 05 的
  20s 存活为 favorable run。**遗留问题, 单独任务跟进** (见后续计划)

## 根因分析
- 碰撞体不对称/缺旋转源自原始 CAD→URDF 导出 (SW2URDF); MuJoCo 转换时髋碰撞为团队自加,
  腿盒旋转遗漏 → 与源不一致。

## 验证方法
1. `mujoco.MjModel.from_xml_path` 编译 + 碰撞体计数/名称检查
2. 碰撞体世界 x 轴方向确认 euler 生效
3. LQR P1 回归 (轮地接触不受影响)
4. 门禁全绿

## 后续计划
- **P4 粗糙地形稳定性**: 单独排查 (spawn 平坦化 / 命令阶段处理 / 地形高度补偿), 需用户确认优先级
- 请用户用 `uv run tools/mujoco/show_collision_model.py` 观察对齐后的碰撞体
- 若修轮 53mm 不对称 (原始 CAD 即如此), 需重新评估影响

## 关联日志
- [[01_collision_model_viewer]] 碰撞体显示 + 轮不对称发现
