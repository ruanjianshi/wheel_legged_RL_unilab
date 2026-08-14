"""LQR / LQI 增益设计 (离散时间, 倒立摆矢状面).

P1 平衡: u = −K·x,  x = [θ, θ̇, v, xpos]
P2+ 指令: LQI 积分增广 z=∫(v−v_ref)dt,  u = −K_aug·[θ, θ̇, v, xpos, z]
"""

from __future__ import annotations

import numpy as np


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """离散 LQR: 迭代求解代数 Riccati 方程, 返回增益 K (u = −K·x)."""
    P = Q.copy()
    for _ in range(2000):
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        if np.max(np.abs(P_new - P)) < 1e-12:
            P = P_new
            break
        P = P_new
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    return np.asarray(K, dtype=np.float64)


def build_lqi(
    A_d: np.ndarray,
    B_d: np.ndarray,
    q_theta: float,
    q_theta_dot: float,
    q_v: float,
    q_x: float,
    q_z: float,
    r: float,
) -> np.ndarray:
    """LQI 积分增广: A_aug = [[A_d,0],[0,1,0,0,0]], 状态 [θ,θ̇,v,xpos,z], ż=v."""
    nx = A_d.shape[0]  # 4
    A_aug = np.zeros((nx + 1, nx + 1), dtype=np.float64)
    A_aug[:nx, :nx] = A_d
    A_aug[nx, 2] = 1.0  # ż = v
    B_aug = np.zeros((nx + 1, 1), dtype=np.float64)
    B_aug[:nx] = B_d
    Q_aug = np.diag([q_theta, q_theta_dot, q_v, q_x, q_z]).astype(np.float64)
    R = np.asarray([[r]], dtype=np.float64)
    return dlqr(A_aug, B_aug, Q_aug, R)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.classic_control.dynamics import build_discrete

    alpha, beta = 3.0, -1.0  # 占位: 真实值用 dynamics.py 测量
    _, _, A_d, B_d = build_discrete(alpha, beta, 0.01)
    K = dlqr(A_d, B_d, np.diag([200, 20, 1, 0.05]), np.asarray([[1.0]]))
    print("K (4-state) =", K)
