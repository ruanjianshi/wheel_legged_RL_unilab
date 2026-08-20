"""八任务评估注册表 (CLAUDE.md §3.1 / §7.1).

每个任务映射: env 注册名 / conf 配置路径 / 训练日志根 / algo / ctrl_dt /
actor 观测维度 / §7.x 达标阈值 (verify.py 判定依据)。

- 8 个 xqrobotwl 任务, 任务间完全独立 (各自 env/conf/shell/devlog/video), 支持并行评估。
- 阈值来自 CLAUDE.md §7.2-7.9 + 附录 A; 指标名与 eval/<task>.py 产出对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # 仓库根


@dataclass(frozen=True)
class Threshold:
    """达标阈值: 指标 op 值 (如 tracking_rmse < 0.1)."""

    op: str  # "<" / "<=" / ">" / ">=" / "==" / "≈"
    value: float
    unit: str = ""


@dataclass
class TaskDef:
    """一个评估任务定义."""

    key: str  # 任务 key, 如 "walk_flat"
    name: str  # 中文名
    env_name: str  # registry env 注册名
    cfg_path: str  # conf mujoco.yaml 相对路径
    log_root: str  # logs 根 (logs/rsl_rl_<algo>/<TaskName>)
    algo: str  # 训练算法 (ppo/cpo/np3o)
    ctrl_dt: float  # 控制周期
    obs_dim: int | None  # actor 观测维度 (None → 运行时从 env 取)
    num_actions: int = 8
    long_eval: str = ""  # §7.0 长时评估要求
    thresholds: dict[str, Threshold] = field(default_factory=dict)


_TASKS: dict[str, TaskDef] = {}


def register(task: TaskDef) -> None:
    _TASKS[task.key] = task


def get(key: str) -> TaskDef:
    if key not in _TASKS:
        raise KeyError(f"未知任务: {key}, 可选: {list(_TASKS)}")
    return _TASKS[key]


def list_tasks() -> dict[str, TaskDef]:
    return dict(_TASKS)


# ── 八任务注册 (阈值按 CLAUDE.md §7.x + 附录 A) ──────────────────────────────

register(
    TaskDef(
        key="walk_flat",
        name="平地滚动行走",
        env_name="XqRobotWLWalkFlat",
        cfg_path="conf/ppo/task/xqrobotwl_walk_flat/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLWalkFlat",
        algo="ppo",
        ctrl_dt=0.01,
        obs_dim=297,
        long_eval="行走 ≥30s",
        thresholds={
            "vx_tracking_rmse": Threshold("<", 0.1, "m/s"),
            "vy_tracking_rmse": Threshold("<", 0.1, "m/s"),
            "survival_rate": Threshold(">=", 0.95, ""),
            "side_vy": Threshold(">=", 0.25, "m/s"),  # §7.2 侧移能力
            "stand_linvel_xy": Threshold("<", 0.2, "m/s"),  # §1.4 微动平衡
            "stand_gyro": Threshold("<", 1.0, "rad/s"),
        },
    )
)

register(
    TaskDef(
        key="toe_walk",
        name="点足平地行走",
        env_name="XqRobotWLToeWalkFlat",
        cfg_path="conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLToeWalkFlat",
        algo="ppo",
        ctrl_dt=0.01,
        obs_dim=306,
        long_eval="行走 ≥30s",
        thresholds={
            "base_height_err": Threshold("<", 0.05, "m"),  # §7.3 机身高度≈0.52±0.05
            "leg_jerk": Threshold("<", 3.0, "rad/s³"),  # §7.3 抬腿平缓
            "vx_tracking_rmse": Threshold("<", 0.15, "m/s"),
            "survival_rate": Threshold(">=", 0.90, ""),
        },
    )
)

register(
    TaskDef(
        key="toe_walk_mode",
        name="双模式点足行走 (站立⇄抬腿)",
        env_name="XqRobotWLToeWalkMode",
        cfg_path="conf/ppo/task/xqrobotwl_toe_walk_mode/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLToeWalkMode",
        algo="ppo",
        ctrl_dt=0.01,
        obs_dim=315,
        long_eval="站立≥10s / 抬腿行走≥30s",
        thresholds={
            "base_height_err": Threshold("<", 0.05, "m"),  # 机身高度≈0.52±0.05
            "vx_tracking_rmse": Threshold("<", 0.15, "m/s"),
            "survival_rate": Threshold(">=", 0.90, ""),
        },
    )
)

register(
    TaskDef(
        key="walk_rough",
        name="不平坦地形行走",
        env_name="XqRobotWLWalkRough",
        cfg_path="conf/ppo/task/xqrobotwl_walk_rough/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLWalkRough",
        algo="ppo",
        ctrl_dt=0.01,
        obs_dim=288,
        long_eval="行走 ≥30s",
        thresholds={
            "survival_rate": Threshold(">=", 0.90, ""),  # §7.4 粗糙地形存活率
            "base_height_std": Threshold("<", 0.06, "m"),  # 机身高度波动小
            "stand_linvel_xy": Threshold("<", 0.25, "m/s"),
            "stand_gyro": Threshold("<", 1.2, "rad/s"),
        },
    )
)

register(
    TaskDef(
        key="jump",
        name="平地跳跃",
        env_name="XqRobotWLJumpFlat",
        cfg_path="conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLJumpFlat",
        algo="ppo",
        ctrl_dt=0.01,
        obs_dim=297,
        long_eval="重复跳跃 ≥10 次",
        thresholds={
            "success_rate": Threshold(">=", 0.90, ""),  # §7.5 跳跃成功率≥90%
            "jump_height": Threshold(">", 0.20, "m"),  # 跳出明显高度
            "air_frac": Threshold(">", 0.05, ""),  # 有腾空
        },
    )
)

register(
    TaskDef(
        key="backflip",
        name="平地后空翻",
        env_name="XqRobotWLBackflipFlat",
        cfg_path="conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLBackflipFlat",
        algo="ppo",
        ctrl_dt=0.005,
        obs_dim=324,
        long_eval="重复 ≥10 次",
        thresholds={
            "flip_rate": Threshold(">=", 0.90, ""),  # §7.6 翻转完成率≥90%
            "land_survival": Threshold(">=", 0.90, ""),  # 落地后稳定站立
        },
    )
)

register(
    TaskDef(
        key="single_leg",
        name="平地单腿平衡 (三态)",
        env_name="XqRobotWLSingleLegFlat",
        cfg_path="conf/ppo/task/xqrobotwl_single_leg_flat/mujoco.yaml",
        log_root="logs/rsl_rl_ppo/XqRobotWLSingleLegFlat",
        algo="ppo",
        ctrl_dt=0.01,
        obs_dim=333,
        long_eval="保持 ≥10s / 行走 ≥30s",
        thresholds={
            "hold_time": Threshold(">=", 5.0, "s"),  # §7.7-A 单腿保持≥5s
            "vx_tracking_rmse": Threshold("<", 0.2, "m/s"),  # §7.7-B 倾斜行走追踪
        },
    )
)

register(
    TaskDef(
        key="fall_recovery",
        name="跌倒恢复",
        env_name="XqRobotWLFallRecoveryFlat",
        cfg_path="conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml",
        log_root="logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat",
        algo="cpo",
        ctrl_dt=0.01,
        obs_dim=297,
        long_eval="每姿态 ≥20 episodes",
        thresholds={
            "recovery_rate": Threshold(">=", 0.80, ""),  # §7.8 恢复率≥80%
            "longest_stand": Threshold(">=", 0.5, "s"),  # 附录A 最长连续站立≥0.5s
            "drift": Threshold("<", 0.5, "m"),  # 附录A 水平漂移<0.5m
            "stand_gyro": Threshold("<", 1.0, "rad/s"),
            "wheel_off_rate": Threshold("<", 0.01, ""),  # 站立期轮子离地率≈0
            # ★ 转圈: 站立期 yaw 累计 (附录A ≈walk 水平~56°=0.98rad; v7 实测63°判为≈walk, 异常线放宽到 1.2rad≈69°)
            "yaw_accum": Threshold("<", 1.2, "rad"),
            # ★ 轮速差 (附录A 小; v7 达标判据 <5 rad/s)
            "wheel_speed_diff": Threshold("<", 5.0, "rad/s"),
        },
    )
)

register(
    TaskDef(
        key="stairs",
        name="抬腿上台阶",
        env_name="XqRobotWLStairs",
        cfg_path="conf/np3o/task/xqrobotwl_stairs/mujoco.yaml",
        log_root="logs/rsl_rl_np3o/XqRobotWLStairs",
        algo="np3o",
        ctrl_dt=0.01,
        obs_dim=297,
        long_eval="每高度 ≥10 次",
        thresholds={
            "success_rate": Threshold(">=", 0.90, ""),  # §7.9 上台阶成功率≥90%
            "survival_rate": Threshold(">=", 0.90, ""),
        },
    )
)
