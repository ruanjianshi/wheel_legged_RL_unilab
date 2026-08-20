# 22 — v10 单模式点足抬腿 (xqrobotwl_toe_walk_flat): 移植双模式验证的机制 + 重训

## 日期
2026-08-19

## 来源
老板拍板: 放弃双模式(站立⇄抬腿同策略), 训练**单模式点足抬腿**, 指定 conf/ppo/task/xqrobotwl_toe_walk_flat。

## 背景 (为什么回单模式)
v6.1-v9 四轮训练实证: 同策略无法稳定承载"静态站立+动态抬腿"双稳态 (v8 站立好腿没, v9 腿有站立塌)。单模式抬腿 = 干净问题域, 把验证过的机制全部用上。

## 修改了什么

| 文件 | 改动 |
|------|------|
| `src/.../toe_walk.py` | ① 窗级交替状态机 `_update_lift_windows` (L/R 摆动窗结束结算: 离地且链完整→奖占比, 未离地→罚) ② `lift_symmetry` 对称 EMA ③ `_tracking_gated_by_command` (零指令不白拿) ④ phase_knee_lift 窗内离地门控 (消灭深弯膝刷分) ⑤ 终止放宽: thigh ±0.9 (原 ±0.5), 接触 8N, 异常帧延迟 5 帧防单帧误杀 ⑥ reward_config 加 symmetry_decay |
| `conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml` | v10 配方: phase_swing_lift 30 (逐时密集奖) + window_penalty 500 (窗级罚) + lift_symmetry -20 + tracking 10/5 (门控) + curriculum_steps 4000 (先抬腿后追踪) + action_scale 0.18/init_noise 0.3 |

## 预期效果
- 双侧交替抬腿成为唯一有利解 (窗级罚 + 密集奖)
- 追踪: 有指令追 (权重 10), 无指令不白拿
- 抬腿探索不被单帧阈值误杀 (终止放宽 + 延迟)

## 验证方法
- 训练曲线: Stage2 (warmup 4000 后) phase_swing_lift 上升 + window_penalty 下行 + lift_sym 低
- 训练后: verify_toe_walk_symmetry.py 交替/对称判定 + 指令追踪实测 + 渲染视频

## 后续计划
- 训练 (GPU1, 1024envs×10k) → 验收 → 视频 → 备份 toe_walk_v10
- 达标后可与"站立专家"自由组合 (双专家架构备用)

## 关联日志
- [21_v9_conclusion_duexpert](2026-08-19/21_v9_conclusion_duexpert.md) — 双模式结论 (单策略双稳态不可能)