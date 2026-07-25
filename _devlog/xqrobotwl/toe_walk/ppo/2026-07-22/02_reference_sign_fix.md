# 02 — 参考轨迹符号修复 (L_knee + R_thigh)

## 日期
2026-07-22

## 来源
评估点足行走模型，发现抬腿幅度太小，追踪根本原因。

## 问题
1. **L_knee 符号反**: `DEFAULT - lift × scale × 5` → 摆动相伸膝(放低轮子)
   正确: `DEFAULT + lift × scale × 5` → 屈膝(抬轮)

2. **R_thigh 符号反**: `DEFAULT + swing × scale × 0.8` → 摆动相后仰
   正确: `DEFAULT - swing × scale × 0.8` → 前倾(R负=前倾)

## 修复
```python
# L_knee: + = bend
ref[:, 2] = DEFAULT[2] + left_lift * scale * 5

# R_knee: - = bend
ref[:, 5] = DEFAULT[5] - right_lift * scale * 5

# R_thigh: - = forward
ref[:, 4] = DEFAULT[4] - right_swing * scale * 0.8
```
L_thigh 和 hip_roll 符号正确，不变。

## 关联日志
- 01 — 点足行走初始训练
