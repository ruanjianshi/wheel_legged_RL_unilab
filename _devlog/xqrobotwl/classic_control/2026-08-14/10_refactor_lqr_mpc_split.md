# [10] ★ 任务轨拆分: LQR 与 MPC 独立 (遵循开发规范 §3 互不干涉)

**日期**: 2026-08-14
**状态**: 完成 — 两轨独立代码/配置/devlog/视频, 回归 P1-P3 100% 与拆分前一致
**关联**: [[../classic_lqr/2026-08-14/02_balance_solved_control_direction]] (LQR), [[../classic_mpc/2026-08-14/09_mpc_completed]] (MPC)

---

## 来源
用户要求"把 LQR 和 MPC 分开写, 遵循开发规范的任务之间互不干涉"。
原 `scripts/classic_control/` 中 LQR/MPC 混在一个 `BalanceController(kind=)` 类 + 共享
`conf/lqr/config.yaml + conf/mpc/config.yaml` (q_theta 既是 LQR 增益又是 MPC 代价, 调一个影响另一个)。

## 修改了什么

### 代码拆分 (scripts/classic_control/)
```
├── common/            # ★ 共享只读基础设施 (开发规范 §3.2)
│   ├── config.py      #   共享加载器 (conf/lqr + conf/mpc, 各轨自取)
│   ├── dynamics.py leg_control.py rollout.py metrics.py report.py render.py
│   ├── base.py        #   BaseController 公共 act() (腿目标/偏航/动作映射)
│   └── run.py eval.py play.py   # 共享运行/评估/交互逻辑
├── lqr/               # ★ LQR 独立轨
│   ├── config.py      #   LqrConfig (conf/lqr, 自包含)
│   ├── controller.py  #   LqrController (独立类)
│   ├── balance_lqr.py eval_lqr.py play_lqr.py render_balance_demo.py lqr.py
└── mpc/               # ★ MPC 独立轨
    ├── config.py      #   MpcConfig (conf/mpc, 自包含)
    ├── controller.py  #   MpcController (独立类)
    ├── qp.py          #   Hildreth QP 求解器
    ├── identify_plant.py  # 黑箱模型辨识 (只读复用 LQR 闭环采集)
    ├── balance_mpc.py eval_mpc.py play_mpc.py
```

### 配置拆分 (conf/)
- `conf/lqr/` 与 `conf/mpc/` — 各轨完整自包含 (robot.yaml + commands.yaml + task/ + config.yaml)
- `conf/lqr/config.yaml` — LQR 权重 (sagittal 直接作增益)
- `conf/mpc/config.yaml` — MPC 权重 (sagittal 作代价 + mpc 模型/约束段)

### 日志/交付拆分
- `_devlog/xqrobotwl/classic_lqr/` (01-05/08) + `classic_mpc/` (06/09) + 共享 07/10
- `video/lqr/` + `video/mpc/`
- `shell/xqrobotwl/classic_lqr/` + `classic_mpc/` (eval/play 独立脚本)

## 哪些文件
- 新增: common/{config,base,run,eval,play}.py, lqr/{config,controller,balance_lqr,eval_lqr,play_lqr}.py,
  mpc/{config,controller,qp,balance_mpc,eval_mpc,play_mpc,identify_plant}.py
- 迁移+改 import: dynamics/leg_control/rollout/metrics/report/render → common/, lqr.py → lqr/
- 删除: 原 config.py/controller.py/run.py/eval_classic.py/play_classic.py + conf/classic_control/
- 备份: `backup/classic_control_pre_split/`

## 训练后效果 (回归验证, 5 ep, 与拆分前完全一致)
| 阶段 | LQR | MPC |
|---|---|---|
| P1 平衡 | 100%, gyro 0.455 | 100%, gyro 0.228 |
| P2 指令 | 100%, vx_rmse 0.080 | 100%, vx_rmse 0.087 |
| P3 腿长 | 100%, height_err 0.038 | 100%, 0.039 |
| P4 地形 | 0% (rough 过难, 遗留) | 0% (并列) |

黑箱模型重新生成 (identify_plant): 留出 RMSE θ 0.0016, v 0.0054 (与重构前一致)。

## 参数调整好坏
- 无参数调整 — 纯结构重构, 数值与拆分前一致。
- **隔离收益**: 调 LQR 权重只改 `conf/lqr`, 不碰 MPC; MPC 的 mpc_horizon/
  integral_gain 等字段 LQR 完全没有 (dataclass 各自独立)。

## 根因分析
原共享 config 中 q_theta 等权重对 LQR (增益) 与 MPC (代价) 语义不同, 共享必然互相牵制;
原 `BalanceController(kind=)` 一个类双算法, 改动互相影响 → 违反 §3 任务隔离。

## 验证方法
1. 构造冒烟: LqrController/MpcController 各 P1/P2 构造 + config 隔离测试
   (改 LQR q_theta=500 不影响 MPC q_theta=100)。
2. 官方 eval (lqr/eval_lqr.py + mpc/eval_mpc.py) 5 ep 存活率与拆分前一致。
3. ruff format + check 通过; 无残留旧路径引用。

## 后续计划
- P4 rough 地形 (两轨并列 0%, 环境过难, 遗留)
- 论文 LQR vs MPC 对比表

## 关联日志
- [[07_refactor_scripts_conf]] 前一次重构 (tools→scripts, conf 分离)
- [[../classic_lqr/INDEX.md]] LQR 轨
- [[../classic_mpc/INDEX.md]] MPC 轨
