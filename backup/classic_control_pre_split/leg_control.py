"""腿长控制 + 地形自适应 (per-leg 高度伺服, 标准 scene_flat.xml position 腿).

运动学复用 vmc.py 标定常数: 2 连杆 (hip_pitch, knee) → 虚拟腿 (L0, theta0).
P3 平地对称高度伺服; P4 粗糙地形 per-leg (wheel_z 反馈).
"""

from __future__ import annotations

import numpy as np

from scripts.classic_control.config import (
    IK_MARGIN,
    JOINT_LIMITS,
    LEG_TARGETS_COMPENSATED,
    STANDING_ANGLES,
    VMC,
)

L1 = VMC["l1"]
L2 = VMC["l2"]
HIP_SIGN = np.asarray(VMC["hip_sign"], dtype=np.float64)  # [L, R]
KNEE_SIGN = np.asarray(VMC["knee_sign"], dtype=np.float64)
C1 = VMC["c1"]
C2 = VMC["c2"]
L0_MIN = VMC["l0_min"] + 0.01
L0_MAX = VMC["l0_max"] - 0.01


def fk_leg(q_hip_pitch: float, q_knee: float, leg: int) -> tuple[float, float, float]:
    """正运动学: (q_pitch, q_knee) → (end_x, end_y, L0). leg∈{0,1} (L,R)."""
    t1 = HIP_SIGN[leg] * q_hip_pitch + C1
    t2 = KNEE_SIGN[leg] * q_knee + C2
    ex = VMC["offset"] + L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    ey = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    L0 = float(np.sqrt(ex * ex + ey * ey))
    return ex, ey, L0


def leg_ik(L0_des: float, theta0_des: float, q0: np.ndarray, leg: int) -> tuple[float, float]:
    """逆运动学: 给定 (L0_des, theta0_des), Newton 5 迭代解 (q_pitch, q_knee).

    theta0 = atan2(end_y, end_x) − π/2 → 目标端点 angle = theta0 + π/2.
    从当前关节角 q0 (2D [q_pitch, q_knee]) 出发。
    """
    ang = theta0_des + np.pi / 2.0
    tx = L0_des * np.cos(ang)
    ty = L0_des * np.sin(ang)
    q = np.asarray(q0, dtype=np.float64).copy()
    for _ in range(5):
        t1 = HIP_SIGN[leg] * q[0] + C1
        t2 = KNEE_SIGN[leg] * q[1] + C2
        ex = VMC["offset"] + L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
        ey = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
        # Jacobian d(end)/d(t1,t2)
        J = np.array(
            [
                [-L1 * np.sin(t1) - L2 * np.sin(t1 + t2), -L2 * np.sin(t1 + t2)],
                [L1 * np.cos(t1) + L2 * np.cos(t1 + t2), L2 * np.cos(t1 + t2)],
            ],
            dtype=np.float64,
        )
        err = np.array([ex - tx, ey - ty], dtype=np.float64)
        if np.max(np.abs(err)) < 1e-6:
            break
        dt1, dt2 = np.linalg.solve(J, -err)
        q[0] += HIP_SIGN[leg] * dt1  # d(theta1) = HIP_SIGN·d(q_pitch)
        q[1] += KNEE_SIGN[leg] * dt2
    return _clamp(q[0], leg, "pitch"), _clamp(q[1], leg, "knee")


def _clamp(val: float, leg: int, kind: str) -> float:
    name = (
        f"{'L' if leg == 0 else 'R'}_hip_pitch"
        if kind == "pitch"
        else f"{'L' if leg == 0 else 'R'}_knee"
    )
    lo, hi = JOINT_LIMITS[name]
    return float(np.clip(val, lo + IK_MARGIN, hi - IK_MARGIN))


def height_targets(
    phase: int,
    h_cmd: float,
    base_z: float,
    base_z_dot: float,
    wheel_z: np.ndarray,
    theta0: np.ndarray,
    joints: np.ndarray,
    params: dict,
) -> np.ndarray:
    """每腿期望 L0.

    phase 3 (平地对称): L_vert_des = L_vert_cur + kp·(h_cmd − base_z) − kd·ż
    phase 4 (地形 per-leg): L_vert_des_i = (base_z − wheel_z_i) + kp·(h_cmd − base_z)
    返回 L0_des[2], clamp 到 [L0_MIN, L0_MAX].
    """
    kp = float(params.get("height_kp", 1.5))
    kd = float(params.get("height_kd", 0.5))
    err = h_cmd - base_z
    dv = -kd * base_z_dot
    l0_des = np.zeros(2, dtype=np.float64)
    for i in range(2):
        l_vert_cur = base_z - float(wheel_z[i])
        l_vert_des = l_vert_cur + kp * err + dv
        if phase >= 4:
            # per-leg: 直接以"轮下地面"为目标保持水平 — 等价上式 (wheel_z 已含地面高低)
            pass
        c_th0 = max(abs(np.cos(float(theta0[i]))), 0.35)
        l0_des[i] = np.clip(l_vert_des / c_th0, L0_MIN, L0_MAX)
    return l0_des


def joints_from_l0(
    l0_des: np.ndarray, theta0_ref: np.ndarray, joints_cur: np.ndarray, hips_roll: np.ndarray
) -> np.ndarray:
    """L0_des[2] + theta0_ref[2] → 6 腿关节目标 (env 顺序 [L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee]).

    theta0_ref 用当前实测 theta0 (保持虚拟腿角度不变, 只伸缩)。
    """
    out = np.zeros(6, dtype=np.float64)
    out[0] = hips_roll[0]
    out[3] = hips_roll[1]
    for i in range(2):
        qp, qk = leg_ik(
            float(l0_des[i]), float(theta0_ref[i]), joints_cur[[1 + 3 * i, 2 + 3 * i]], i
        )
        out[1 + 3 * i] = qp
        out[2 + 3 * i] = qk
    return out


def fk_legs(dof_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """实测腿 → 每腿 (L0, theta0). dof_pos 为 8D env 顺序."""
    L0 = np.zeros(2, dtype=np.float64)
    theta0 = np.zeros(2, dtype=np.float64)
    for i in range(2):
        ex, ey, l0 = fk_leg(float(dof_pos[1 + 3 * i]), float(dof_pos[2 + 3 * i]), i)
        L0[i] = l0
        theta0[i] = float(np.arctan2(ey, ex)) - np.pi / 2.0
    return L0, theta0


def standing_joint_targets() -> np.ndarray:
    """P1/P2 腿目标 = 下垂补偿后的站姿伺服目标 (6D env 顺序).

    kp=60 在负重下膝部下垂 ~0.22 rad, 命令补偿值使实际关节 ≈ 自然站姿。
    """
    return np.asarray(LEG_TARGETS_COMPENSATED, dtype=np.float64)
