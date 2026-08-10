"""评估指标 (CLAUDE.md 附录 A 核心指标 + §7.x 追踪/稳定/运动质量).

数据载体: 每步一个 ``StepSample`` (engine 逐行采集), 一个 episode 一个 ``Trace``。
指标函数分两类:
  - 单 trace 指标: 作用于一条 episode 的时间序列 (站立高度/漂移/gyro/轮速…)
  - 多 trace 聚合指标: 作用于一组 episode (恢复率/保持率/成功率…)

附录 A 定义 (达标值):
  恢复率≥80% · 最长连续站立≥0.5s · 站立高度≈0.52m · 水平漂移<0.5m
  yaw 累计≈56°(walk) · |gyro|<1 rad/s · 轮速差小 · 轮子离地率 0%
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StepSample:
    """单步完整姿态采样 (与 dump_pose_data 的 26 列 CSV 对齐)."""

    step: int
    time_s: float
    dof_pos: np.ndarray  # (6,) 腿部关节角
    euler: np.ndarray  # (3,) roll/pitch/yaw (rad)
    base_pos: np.ndarray  # (3,) base 世界位置
    linvel: np.ndarray  # (3,) 本地线速度
    gyro: np.ndarray  # (3,) 本地角速度
    wheel_vel: np.ndarray  # (2,) 轮子角速度
    up_z: float  # 直立度
    wheel_contact: np.ndarray  # (2,) 轮地接触 0/1
    recover_completed: float = 0.0  # 恢复完成锁存 0/1
    cmd: np.ndarray | None = None  # 5D 命令 (可选)

    def to_row(self, col_idx: dict[str, int]) -> list:
        """转成 dump_pose_data 的 26 列 CSV 行 (2 位小数)."""
        row = np.zeros(26)
        row[col_idx["step"]] = self.step
        row[col_idx["time_s"]] = self.time_s
        row[col_idx["L_hip_roll_rad"] : col_idx["L_hip_roll_rad"] + 6] = self.dof_pos
        row[col_idx["base_roll_rad"] : col_idx["base_roll_rad"] + 3] = self.euler
        row[col_idx["base_x"] : col_idx["base_x"] + 3] = self.base_pos
        row[col_idx["linvel_x"] : col_idx["linvel_x"] + 3] = self.linvel
        row[col_idx["gyro_x"] : col_idx["gyro_x"] + 3] = self.gyro
        row[col_idx["wheel_L_rads"] : col_idx["wheel_L_rads"] + 2] = self.wheel_vel
        row[col_idx["up_z"]] = self.up_z
        row[col_idx["wheel_contact_L"]] = int(self.wheel_contact[0])
        row[col_idx["wheel_contact_R"]] = int(self.wheel_contact[1])
        row[col_idx["recover_completed"]] = int(self.recover_completed)
        return [round(v, 2) if i not in (col_idx["step"],) else int(v) for i, v in enumerate(row)]


Trace = list[StepSample]


# ── 站立帧判定 (§1.3/§1.4: z>0.45 + 直立>0.85 + 双轮着地) ─────────────────────


def upright_mask(trace: Trace) -> np.ndarray:
    """直立帧掩码: base_z 正常 + 直立度>0.85 (通用站立, 不依赖轮地接触).

    walk 类 env 不填充 wheel_contact, 故站立判定不能依赖轮地信号。
    """
    z = np.array([s.base_pos[2] for s in trace])
    up = np.array([s.up_z for s in trace])
    return (z > 0.45) & (z < 0.65) & (up > 0.85)


def standing_mask(trace: Trace) -> np.ndarray:
    """稳定站立帧掩码 (附录 A 定义: z>0.45 + 直立>0.85 + 双轮着地).

    用于 fall_recovery 等会填充 wheel_contact 的任务。
    """
    if not trace or not np.any(np.array([s.wheel_contact for s in trace]) > 0.5):
        return np.zeros(len(trace), dtype=bool)
    z = np.array([s.base_pos[2] for s in trace])
    up = np.array([s.up_z for s in trace])
    wc = np.array([s.wheel_contact for s in trace])
    return (z > 0.45) & (up > 0.85) & (np.min(wc, axis=1) > 0.5)


def _arr(trace: Trace, key: str) -> np.ndarray:
    if key == "z":
        return np.array([s.base_pos[2] for s in trace])
    if key == "x":
        return np.array([s.base_pos[0] for s in trace])
    if key == "y":
        return np.array([s.base_pos[1] for s in trace])
    if key == "up":
        return np.array([s.up_z for s in trace])
    if key == "yaw":
        return np.array([s.euler[2] for s in trace])
    if key == "gyro_mag":
        return np.array([np.linalg.norm(s.gyro) for s in trace])
    if key == "linvel_xy":
        return np.array([np.hypot(s.linvel[0], s.linvel[1]) for s in trace])
    if key == "wheel_diff":
        return np.array([abs(s.wheel_vel[0] - s.wheel_vel[1]) for s in trace])
    raise KeyError(key)


def _unwrap(vals: np.ndarray) -> np.ndarray:
    out = np.empty_like(vals)
    prev = 0.0
    for i, v in enumerate(vals):
        if i == 0:
            out[i] = v
        else:
            d = v - prev
            d = (d + math.pi) % (2.0 * math.pi) - math.pi
            out[i] = out[i - 1] + d
        prev = v
    return out


# ── 单 trace 指标 ────────────────────────────────────────────────────────────


def stand_height(trace: Trace) -> float:
    """站立期平均 base_z (附录A 站立高度, 目标≈0.52m)."""
    m = upright_mask(trace)
    if not m.any():
        return float("nan")
    return float(np.mean(_arr(trace, "z")[m]))


def stand_height_err(trace: Trace, target: float = 0.52) -> float:
    """站立高度偏离目标值 (§7.3 机身高度≈0.52±0.05)."""
    h = stand_height(trace)
    return float("nan") if math.isnan(h) else abs(h - target)


def stand_height_std(trace: Trace) -> float:
    """站立期 base_z 标准差 (§7.4 机身高度波动小)."""
    m = upright_mask(trace)
    return float(np.std(_arr(trace, "z")[m])) if m.sum() > 1 else float("nan")


def drift(trace: Trace) -> float:
    """水平漂移: episode 内 base_x 相对起点的最大偏移 (附录A <0.5m)."""
    x = _arr(trace, "x")
    if len(x) == 0:
        return float("nan")
    return float(np.max(np.abs(x - x[0])))


def yaw_accum(trace: Trace) -> float:
    """站立期 yaw 累计旋转 (附录A, walk 水平≈56°=0.98rad)."""
    m = upright_mask(trace)
    if m.sum() < 2:
        return 0.0
    yaw = _unwrap(_arr(trace, "yaw")[m])
    return float(abs(yaw[-1] - yaw[0]))


def mean_gyro(trace: Trace) -> float:
    """站立期平均 |gyro| (附录A <1 rad/s)."""
    m = upright_mask(trace)
    return float(np.mean(_arr(trace, "gyro_mag")[m])) if m.any() else float("nan")


def mean_linvel_xy(trace: Trace) -> float:
    """站立期平均 |linvel_xy| (§1.4 微动平衡 <0.2 m/s)."""
    m = upright_mask(trace)
    return float(np.mean(_arr(trace, "linvel_xy")[m])) if m.any() else float("nan")


def wheel_speed_diff(trace: Trace) -> float:
    """平均左右轮速差 |wL-wR| (附录A 轮速差小)."""
    return float(np.mean(_arr(trace, "wheel_diff"))) if trace else float("nan")


def wheel_off_rate(trace: Trace) -> float:
    """直立期轮子离地率 (附录A =0); env 未填充轮地接触时返回 NaN."""
    if not trace or not np.any(np.array([s.wheel_contact for s in trace]) > 0.5):
        return float("nan")
    m = upright_mask(trace)
    if not m.any():
        return 1.0
    wc = np.array([s.wheel_contact for s in trace])[m]
    off = np.mean(np.min(wc, axis=1) < 0.5)
    return float(off)


def longest_stand_s(trace: Trace, ctrl_dt: float) -> float:
    """最长连续站立时长 (附录A ≥0.5s)."""
    m = standing_mask(trace)
    best = cur = 0
    for flag in m:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return float(best * ctrl_dt)


def double_wheel_on_rate(trace: Trace) -> float:
    """双轮着地帧占比."""
    if not trace:
        return float("nan")
    wc = np.array([s.wheel_contact for s in trace])
    return float(np.mean(np.min(wc, axis=1) > 0.5))


def mean_max_z(trace: Trace) -> float:
    """平均最大 base_z (恢复任务: 躺地~0.15, 站立~0.55)."""
    return float(np.max(_arr(trace, "z"))) if trace else float("nan")


def mean_max_up(trace: Trace) -> float:
    """平均最大直立度."""
    return float(np.max(_arr(trace, "up"))) if trace else float("nan")


def leg_jerk(trace: Trace, ctrl_dt: float = 0.01) -> float:
    """抬腿平缓度: 腿部关节角的平均加加速度 (越低越平缓, §7.3)."""
    if len(trace) < 4:
        return float("nan")
    pos = np.array([s.dof_pos for s in trace])  # (N,6)
    acc = np.diff(pos, n=2, axis=0) / (ctrl_dt * ctrl_dt)  # 二阶差分
    jerk = np.diff(acc, axis=0)  # 三阶差分
    return float(np.mean(np.abs(jerk))) if len(jerk) else float("nan")


def action_smoothness(trace: Trace, actions: list[np.ndarray]) -> float:
    """动作平滑度: 相邻动作差 L2 均值 (低=平滑)."""
    if len(actions) < 2:
        return float("nan")
    return float(
        np.mean([np.linalg.norm(actions[i] - actions[i - 1]) for i in range(1, len(actions))])
    )


def leg_symmetry(trace: Trace) -> float:
    """左右腿关节角对称性: mean|L_pos - R_pos| (低=对称)."""
    if not trace:
        return float("nan")
    pos = np.array([s.dof_pos for s in trace])
    return float(np.mean(np.abs(pos[:, :3] - pos[:, 3:6])))


# ── 追踪指标 (需要命令, 逐场景) ──────────────────────────────────────────────


def tracking_rmse(trace: Trace, cmd: np.ndarray) -> dict[str, float]:
    """本地 linvel 对命令的 RMSE (vx/vy) + yaw 率 (gyro_z)."""
    if not trace:
        return {"vx": float("nan"), "vy": float("nan"), "vyaw": float("nan")}
    lv = np.array([s.linvel for s in trace])
    gz = np.array([s.gyro[2] for s in trace])
    return {
        "vx": float(np.sqrt(np.mean((lv[:, 0] - cmd[0]) ** 2))),
        "vy": float(np.sqrt(np.mean((lv[:, 1] - cmd[1]) ** 2))),
        "vyaw": float(np.sqrt(np.mean((gz - cmd[2]) ** 2))),
    }


def avg_linvel(trace: Trace) -> tuple[float, float]:
    """平均本地 vx / vy."""
    if not trace:
        return float("nan"), float("nan")
    lv = np.array([s.linvel for s in trace])
    return float(np.mean(lv[:, 0])), float(np.mean(lv[:, 1]))


def vel_coupling(trace: Trace, cmd: np.ndarray) -> float:
    """Vx-only 时的 Vy 串扰 (或反之): |avg(另一轴)|."""
    vx, vy = avg_linvel(trace)
    if abs(cmd[0]) > 1e-4 and abs(cmd[1]) < 1e-4:
        return abs(vy)
    if abs(cmd[1]) > 1e-4 and abs(cmd[0]) < 1e-4:
        return abs(vx)
    return 0.0


# ── 多 trace 聚合指标 (episode 级) ──────────────────────────────────────────


def recovery_rate(traces: list[Trace], stand_z: float = 0.52) -> float:
    """恢复率: base_z 达到站立高度的 episode 比例 (附录A ≥80%)."""
    if not traces:
        return float("nan")
    ok = sum(1 for t in traces if np.any(_arr(t, "z") > stand_z))
    return ok / len(traces)


def stay_up_rate(traces: list[Trace], max_steps: int) -> float:
    """保持率: 跑到 max_steps 的 episode 比例."""
    if not traces:
        return float("nan")
    return sum(1 for t in traces if len(t) >= max_steps) / len(traces)


def survival_rate(traces: list[Trace], max_steps: int) -> float:
    """存活率: 跑到 max_steps 的 episode 比例 (与保持率同义, 用于行走类)."""
    return stay_up_rate(traces, max_steps)


def task_success_rate(traces: list[Trace], success_fn) -> float:
    """通用成功率: success_fn(trace) → bool 的 episode 占比."""
    if not traces:
        return float("nan")
    return sum(1 for t in traces if success_fn(t)) / len(traces)


def flip_completion(trace: Trace, base_up_vec_axis: str = "euler_roll") -> float:
    """后空翻完成度: base roll 累积转过 360° (2π) 记为 1, 否则比例. (任务自定义用)"""
    if not trace:
        return 0.0
    roll = np.array([s.euler[0] for s in trace])
    drift = _unwrap(roll)
    total = abs(drift[-1] - drift[0]) if len(drift) > 1 else 0.0
    return float(min(total / (2.0 * math.pi), 1.0))


def jump_height(trace: Trace, standing_z: float | None = None) -> float:
    """跳跃高度: 最大 base_z - 站立基准 (trigger off 且双轮着地的 z 中位数)."""
    z = _arr(trace, "z")
    if len(z) == 0:
        return float("nan")
    max_z = float(np.max(z))
    if standing_z is None:
        m = standing_mask(trace)
        baseline = float(np.median(z[m])) if m.any() else float(np.median(z))
    else:
        baseline = standing_z
    return max_z - baseline


def air_frac(trace: Trace) -> float:
    """腾空占比: 轮子离地帧占比 (跳跃/后空翻)."""
    if not trace:
        return float("nan")
    wc = np.array([s.wheel_contact for s in trace])
    return float(np.mean(np.min(wc, axis=1) < 0.5))


# ── 汇总工具 ────────────────────────────────────────────────────────────────


def trace_append_actions(trace: Trace, actions: list[np.ndarray]) -> list[np.ndarray]:
    """把动作序列按采样对齐返回 (与 trace 同长, 供 action_smoothness 用)."""
    return actions
