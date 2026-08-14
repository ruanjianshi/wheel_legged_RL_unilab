# 经典控制轨 (LQR + MPC) — 两条独立任务轨

两轮足机器人 xqrobotwl 的经典(模型基)平衡控制器。按开发规范 §3 任务隔离,
LQR 与 MPC 拆为**两条独立任务轨**, 仅共享只读机器人接口 (互不干涉)。

## 目录结构 (独立任务轨)

```
scripts/classic_control/
├── common/            # ★ 共享只读基础设施 (机器人/环境接口)
│   ├── config.py      #   机器人常数 + 命令表 (conf/classic_common)
│   ├── dynamics.py leg_control.py rollout.py metrics.py report.py render.py
│   ├── base.py        #   BaseController 公共 act() (腿目标/偏航/动作映射)
│   └── run.py eval.py play.py   # 共享运行/评估/交互逻辑
├── lqr/               # ★ LQR 独立轨
│   ├── config.py controller.py balance_lqr.py eval_lqr.py play_lqr.py
└── mpc/               # ★ MPC 独立轨
    ├── config.py controller.py qp.py identify_plant.py
    ├── balance_mpc.py eval_mpc.py play_mpc.py
```

配置各自完全自包含 (仅两个目录, 互不干涉):
```
conf/lqr/             # LQR 轨完整配置
│   ├── robot.yaml    #   机器人常数
│   ├── commands.yaml #   命令表 (yaw/命令/腿控/阶段)
│   ├── task/         #   env 映射 (flat/rough)
│   └── config.yaml   #   LQR 算法权重 (sagittal 直接作增益)
conf/mpc/             # MPC 轨完整配置 (同上 + mpc 模型/约束段)
    ├── robot.yaml commands.yaml task/
    └── config.yaml   #   MPC 权重 (sagittal 作代价 + mpc 段)
```

调参只改各自 conf, 互不影响: 改 `conf/lqr/config.yaml` 不碰 MPC,
改 `conf/mpc/config.yaml` 不碰 LQR。机器人/命令表内容两轨相同 (同一台机器人,
共享只读); 算法权重各自独立。

## 能力 (两轨均已达成 P1-P3)

| 阶段 | 能力 | LQR | MPC |
|---|---|---|---|
| P1 | 稳定平衡 | 12s 存活 ✅ | 12s 存活, gyro 0.228 ✅ |
| P2 | 指令控制 | vx RMSE 0.080 ✅ | vx RMSE 0.093 ✅ |
| P3 | 腿长控制 | height_err 0.038 ✅ | height_err 0.039 ✅ |
| P4 | 地形自适应 | ⚠️ rough 过难 (0%) | ⚠️ 与 LQR 并列 (0%) |

## 评估 (各自独立脚本)

```bash
# LQR
bash shell/xqrobotwl/classic_lqr/eval_lqr.sh all          # 全阶段 5 ep
uv run python scripts/classic_control/lqr/eval_lqr.py --phases 1 2 3 4 --episodes 5
# MPC
bash shell/xqrobotwl/classic_mpc/eval_mpc.sh all
uv run python scripts/classic_control/mpc/eval_mpc.py --phases 1 2 3 4 --episodes 5
```

单阶段: `uv run python scripts/classic_control/lqr/balance_lqr.py --phase 2 --sim_time 40`
MPC: `uv run python scripts/classic_control/mpc/balance_mpc.py --phase 2 --sim_time 35`

## 交互控制 (各自独立)

```bash
bash shell/xqrobotwl/classic_lqr/play_lqr.sh              # LQR 平地
bash shell/xqrobotwl/classic_mpc/play_mpc.sh rough        # MPC 粗糙地形
```
键盘: ↑/↓ 前进后退 · ←/→ 转向 · Q/E 腿长 · Enter 停止 · Space 暂停 · Backspace 重置

## MPC 说明
- 线性 MPC (Hildreth QP, 求解 <1ms), 黑箱模型 `logs/classic/mpc_plant_bb.npz`
  (LQR 闭环+探针辨识, `scripts/classic_control/mpc/identify_plant.py`)。
- 模型来源优先级: CLI A_d/B_d → config `model_file` → 解析 (α,β,τ)。
- P1 用解析模型 (黑箱 P1 漂移不达标); P2+ 用黑箱 (速度跟踪准)。

## 渲染视频
- LQR: `video/lqr/` (lqr_env_balance_SUCCESS 等)
- MPC: `video/mpc/` (mpc_p1_balance / p2_command / p3_height / p4_rough)

## 开发日志 (独立)
- `_devlog/xqrobotwl/classic_lqr/INDEX.md`
- `_devlog/xqrobotwl/classic_mpc/INDEX.md`
- 共享基础设施: `_devlog/xqrobotwl/classic_control/INDEX.md`
