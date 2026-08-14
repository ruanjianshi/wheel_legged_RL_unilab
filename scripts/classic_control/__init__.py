"""经典控制轨 — 共享只读基础设施 + LQR/MPC 两条独立任务轨.

按开发规范 §3 (任务之间互不干涉) 拆分:
- `common/`  共享只读基础设施 (机器人/环境接口: config/dynamics/rollout/leg_control/
            metrics/report/render + BaseController 公共 act + run/eval/play 共享逻辑)
- `lqr/`     LQR 独立任务轨 (config/controller/balance_lqr/eval_lqr/play_lqr)
- `mpc/`     MPC 独立任务轨 (config/controller/qp/identify_plant/balance_mpc/eval_mpc/play_mpc)

各自独立: conf/lqr + conf/mpc (权重互不干涉),
_devlog/xqrobotwl/classic_lqr + classic_mpc (日志互不干涉),
video/lqr + video/mpc (交付互不干涉)。
"""
