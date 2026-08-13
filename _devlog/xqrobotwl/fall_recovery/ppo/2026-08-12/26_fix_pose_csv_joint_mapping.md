# [26] 修复: 姿态数据 CSV 关节列映射错位 (R 腿读到 L_wheel, 假阳性"髋外展")

**日期**: 2026-08-12
**来源**: 用户要求按规范做恢复后站立姿态评估 (§1.3/§1.5) → 导出姿态 CSV 分析时发现
R_hip_roll 值达 -9.94 rad (远超关节限位 ±3.14), 追查定位为数据工具列映射 bug
**关联**: [[24_overnight_v8_v82_delivery]], [[25_fix_cpo_play_support]]

---

## 根因分析

xqrobotwl 关节序 (MuJoCo dof, 与 qpos 差 7 个 free joint):

```
dof: [L_hip_roll, L_hip_pitch, L_knee, L_wheel, R_hip_roll, R_hip_pitch, R_knee, R_wheel]
       idx0         idx1         idx2    idx3     idx4         idx5        idx6    idx7
```

两个数据工具误用连续切片取腿部 6 关节:

| 工具 | 错误代码 | 实际读到 |
|---|---|---|
| `tools/xqrobotwl/dump_pose_data.py` L217 | `dof_pos[0:6]` | idx3=L_wheel 被当成 R_hip_roll → **R_hip_roll 列 = L_wheel 自由累加角 (可达 ±10 rad)** |
| `_devlog/assess/engine.py` L183/188 | `dof_pos[0:6]` + `dof_vel[6:8]` | 同错; 轮速列 `[6:8]` 读到 R_knee/R_wheel, R_hip_pitch 列读到 R_hip_roll |

**后果**: 用 CSV 做 §1.3 站姿推理时, "R_hip_roll" 显示 -2.9 ~ -9.9 rad → 误判"髋外展严重异常"。
实际上该列是 L_wheel 的累计角 (自由轮, 数值无意义)。

## 修改了什么

| 文件 | 改动 |
|---|---|
| `tools/xqrobotwl/dump_pose_data.py` | 腿部 6 关节 `dof_pos[[0,1,2,4,5,6]]`; 轮速 `dof_vel[[3,7]]` |
| `_devlog/assess/engine.py` | 同上 (StepSample.dof_pos / wheel_vel) |

## 验证方法

修复后重新导出 3 姿态 CSV + 站姿分析:

| 姿态 | 站姿 | 关节偏差(max) | 高度 | 直立 | 一前一后 | 摇摆gyro | 轮离地 | yaw |
|---|---|---|---|---|---|---|---|---|
| 俯卧 | **✅ 正常** | 0.169 | 0.519 | 0.999 | 0.14 | 0.41 | 0/434 | 3.4° |
| 左躺 | **✅ 正常** | 0.165 | 0.519 | 0.995 | 0.17 | 0.30 | 16/441 | 40.7° |
| 右躺 | **✅ 正常** | 0.151 | 0.522 | 0.999 | 0.16 | 0.53 | 0/422 | 28.1° |

**结论**: 修复前"髋外展"为假阳性; 修复后 model_7000 恢复后的站立姿态**正常** —
关节贴近自然站姿 (standing_angles), 高度≈0.52m, 直立>0.99, 无摇摆/一前一后, 轮地基本贴合,
yaw 3-41° (≈walk 水平)。左躺有 16/441 步短暂轮离地 (3.6%, 轻微)。

runner 冒烟 (10 env 快速): 指标输出正常, wheel_speed_diff 读到正确轮速。

## 教训与规范意义

- **§1.5 数据优先**: 数据工具的正确性是姿态评估的前提, 列映射必须对关节序有据可查
- 关节序在 env 复位代码 (DR provider qpos 映射) 有明确来源, 切片前应先核对
- 大数值关节角 (> 关节限位) 是数据错误的红旗, 应即时追查而非当作真实姿态

## 后续计划

- [ ] 复核其余任务 (jump/single_leg 等) 是否用了同样的错误切片 (assess engine 是共享的, 已修;
      tools 里其他 dump 类脚本需 grep 核对)
- [ ] 评估报告更新: 恢复后站姿结论修正为"正常" (原 devlog #24 站姿相关表述以本修正为准)
