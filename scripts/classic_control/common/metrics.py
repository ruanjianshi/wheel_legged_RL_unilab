"""从录制轨迹计算 §7.2/§7.4/§7.0 指标 (经典控制轨自包含)."""

from __future__ import annotations

import numpy as np


def _quat_yaw(q: np.ndarray) -> float:
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def compute(record: list[dict], phase: int) -> dict[str, float]:
    if not record:
        return {}
    up_z = np.array([r["sensors"]["up"][2] for r in record])
    gyro = np.array([r["sensors"]["gyro"] for r in record])
    linvel = np.array([r["sensors"]["linvel"] for r in record])
    base_z = np.array([r["sensors"]["base_z"] for r in record])
    theta = np.array([r["sensors"]["theta"] for r in record])
    cmd_vx = np.array([r["cmd"][0] for r in record])
    v = np.array([r["sensors"]["v"] for r in record])

    # 站立掩码 (同 fall_recovery: up_z>0.85 & base_z>0.45)
    standing = (up_z > 0.85) & (base_z > 0.45)
    # 最长连续站立
    max_hold = 0.0
    cur = 0.0
    for s in standing:
        cur = cur + 1.0 if s else 0.0
        max_hold = max(max_hold, cur)
    max_hold_s = max_hold * 0.01  # dt=0.01

    # yaw 累计 (从 qpos 四元数)
    yaw_first = _quat_yaw(record[0]["state"][4:8])
    yaw_last = _quat_yaw(record[-1]["state"][4:8])
    yaw_turn_deg = abs(np.degrees((yaw_last - yaw_first + np.pi) % (2 * np.pi) - np.pi))

    m: dict[str, float] = {
        "ep_len_s": len(record) * 0.01,
        "survived": float(not record[-1]["terminated"]),
        "stand_hold_max_s": max_hold_s,
        "gyro_rms": float(np.sqrt(np.mean(np.sum(gyro**2, axis=1)))),
        "linvel_xy_mean": float(np.mean(np.linalg.norm(linvel[:, :2], axis=1))),
        "base_z_mean": float(np.mean(base_z)),
        "base_z_std": float(np.std(base_z)),
        "yaw_turn_deg": yaw_turn_deg,
        "tilt_rms": float(np.sqrt(np.mean(theta**2))),
    }

    if phase >= 2:
        # ★ §7.2 稳态追踪: 排除命令变化后 1s 的加速瞬态 (命令已恒定>1s 的步)
        moving = np.abs(cmd_vx) > 0.05
        steady_mask = moving.copy()
        for i in range(1, len(cmd_vx)):
            if moving[i] and cmd_vx[i] == cmd_vx[i - 1]:
                pass  # keep
            elif moving[i]:
                # 命令刚变化, 标记接下来 1s (100步) 为瞬态
                for j in range(i, min(i + 100, len(cmd_vx))):
                    if cmd_vx[j] == cmd_vx[i]:
                        steady_mask[j] = False
                    else:
                        break
        if np.any(steady_mask):
            m["vx_rmse"] = float(np.sqrt(np.mean((v[steady_mask] - cmd_vx[steady_mask]) ** 2)))
            m["vx_mean_err"] = float(np.mean(np.abs(v[steady_mask] - cmd_vx[steady_mask])))
        else:
            m["vx_rmse"] = 0.0
            m["vx_mean_err"] = 0.0

    if phase == 3:
        heights = np.unique([round(r["cmd"][4], 3) for r in record])
        errs = []
        for h in heights:
            mask = np.array([abs(r["cmd"][4] - h) < 1e-3 for r in record])
            if np.any(mask):
                errs.append(float(np.mean(np.abs(base_z[mask] - h))))
        m["height_err_mean"] = float(np.mean(errs)) if errs else 0.0

    if phase == 4:
        # 用 knee 变化范围近似腿长自适应
        knee_L = np.array([r["sensors"]["dof_pos"][2] for r in record])
        knee_R = np.array([r["sensors"]["dof_pos"][5] for r in record])
        m["knee_L_range"] = float(np.ptp(knee_L))
        m["knee_R_range"] = float(np.ptp(knee_R))
    return m


def threshold_for(phase: int) -> dict[str, tuple[float, float, str]]:
    """phase → {指标: (阈值, 方向, 单位)}. 方向: '<' 越小越好, '>' 越大越好."""
    if phase == 1:
        return {
            "stand_hold_max_s": (10.0, ">", "s"),
            "gyro_rms": (1.0, "<", "rad/s"),
            "linvel_xy_mean": (0.2, "<", "m/s"),
            "yaw_turn_deg": (30.0, "<", "deg"),
        }
    if phase == 2:
        return {
            "stand_hold_max_s": (10.0, ">", "s"),
            "vx_rmse": (0.1, "<", "m/s"),
            "gyro_rms": (1.0, "<", "rad/s"),
        }
    if phase == 3:
        return {
            "stand_hold_max_s": (10.0, ">", "s"),
            "height_err_mean": (0.05, "<", "m"),
        }
    # phase 4
    return {
        "stand_hold_max_s": (10.0, ">", "s"),
        "tilt_rms": (0.35, "<", "rad"),
        "base_z_std": (0.08, "<", "m"),
    }
