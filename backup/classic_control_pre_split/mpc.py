"""线性 MPC (轮速命令倒立摆) — 手写向量化 Hildreth QP (numpy only).

模型 (w_c convention, rad/s): 状态 x=[θ,θ̇,v,xpos], 控制 u=轮速命令 (rad/s).
  v̇ = (u·R − v)/τ      轮速度伺服一阶滞后
  θ̈ = α·θ + β·v̇        倒立摆 + 轮加速耦合

预测 X = P·x₀ + H·U; 代价 J = UᵀGU + 2·gᵀU (可含 LQR 末端代价);
约束 |u|≤u_max, |θ|≤θ_max, |v|≤v_max → C·U ≤ d + box。

性能 (★ 原瓶颈): build_mpc_matrices 一次预计算 (Ginv/E/C 与 x₀ 无关);
每拍 solve_qp 仅做矩阵-向量乘 + 向量化 Gauss-Seidel (残差跟踪) → 亚毫秒级,
远低于 10ms 控制周期。
"""

from __future__ import annotations

import numpy as np


def dlqr_riccati(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    max_iter: int = 2000,
    tol: float = 1e-10,
) -> np.ndarray:
    """离散 LQR 代价矩阵 P (Riccati 迭代): P = AᵀPA − AᵀPB(BᵀPB+R)⁻¹BᵀPA + Q."""
    P = Q.astype(np.float64).copy()
    for _ in range(max_iter):
        K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
        P_new = A.T @ P @ A - A.T @ P @ B @ K + Q
        if float(np.max(np.abs(P_new - P))) < tol:
            return P_new
        P = P_new
    return P


def build_mpc_matrices(
    A_d: np.ndarray,
    B_d: np.ndarray,
    N: int,
    Q: np.ndarray,
    R: np.ndarray,
    P_term: np.ndarray | None = None,
    r_rate: float = 0.0,
) -> dict:
    """预计算 MPC 常数 (可选 LQR 末端代价 + 控制变化率惩罚). 返回 dict.

    X = P·x₀ + H·U (N 步预测); G 含末端代价 H_Nᵀ·P_term·H_N 与 r_rate·DᵀD;
    Ginv/E/CGinv 与 x₀ 无关 → 每拍仅重算 g/d。
    """
    nx, nu = A_d.shape[0], B_d.shape[1]
    P = np.zeros((N * nx, nx))
    H = np.zeros((N * nx, N * nu))
    A_pow = np.eye(nx)
    for k in range(N):
        P[k * nx : (k + 1) * nx] = A_pow
        A_pow = A_d @ A_pow
    for k in range(N):
        for j in range(k + 1):
            H[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = (
                np.linalg.matrix_power(A_d, k - j) @ B_d
            )
    Q_bar = np.kron(np.eye(N), Q)
    R_bar = np.kron(np.eye(N), R)
    G = H.T @ Q_bar @ H + R_bar
    P_N = H_N = P_term_out = D_rate = None
    if P_term is not None:
        H_N = np.hstack([np.linalg.matrix_power(A_d, N - 1 - j) @ B_d for j in range(N)])
        G = G + H_N.T @ P_term @ H_N
        P_N = np.linalg.matrix_power(A_d, N)
        P_term_out = P_term
    # 控制变化率惩罚: Δ=D·U−u₋₁·e₀ → G += r_rate·DᵀD (g 的线性项在求解时加)
    if r_rate > 0.0:
        D_rate = np.zeros((N, N))
        for i in range(N):
            D_rate[i, i] = 1.0
            if i > 0:
                D_rate[i, i - 1] = -1.0
        G = G + r_rate * (D_rate.T @ D_rate)
    Ginv = np.linalg.inv(G + 1e-8 * np.eye(G.shape[0]))

    # 约束 C·U ≤ d: ±θ_k, ±v_k (N 步), 与 x₀ 无关 (x₀ 只进 d)
    rows = []
    for k in range(N):
        Hk = H[k * nx : (k + 1) * nx]
        rows.append(Hk[0, :])
        rows.append(-Hk[0, :])  # ±θ_k
        rows.append(Hk[2, :])
        rows.append(-Hk[2, :])  # ±v_k
    C = np.array(rows, dtype=np.float64)
    E = C @ Ginv @ C.T
    return {
        "P": P,
        "H": H,
        "G": G,
        "Ginv": Ginv,
        "C": C,
        "E": E,
        "CGinv": C @ Ginv,
        "P_N": P_N,
        "H_N": H_N,
        "P_term": P_term_out,
        "nx": nx,
        "nu": nu,
        "N": N,
        "Q_bar": Q_bar,
    }


def solve_qp(
    mpc: dict,
    g: np.ndarray,
    d: np.ndarray,
    u_max: float,
    lam0: np.ndarray | None = None,
) -> tuple[float, np.ndarray, dict]:
    """一步滚动求解 (预计算 mpc 常数 + 当前 g/d) → (u₀, λ, stats).

    Hildreth 对偶 (min ½λᵀEλ+λᵀkk, λ≥0) 投影 Gauss-Seidel:
      λᵢ ← max(0, λᵢ − ω·(Eλ+kk)ᵢ/Eᵢᵢ),  残差跟踪向量化 (★ 修正符号).
    """
    C, Ginv, E, CGinv = mpc["C"], mpc["Ginv"], mpc["E"], mpc["CGinv"]
    m = E.shape[0]
    kk = CGinv @ g + d
    lam = np.zeros(m, dtype=np.float64) if lam0 is None else lam0.copy()
    res = E @ lam + kk
    iters = 0
    for it in range(300):
        lam_old = lam.copy()
        for i in range(m):
            w = -res[i] / E[i, i]  # 下降方向步长
            new = lam[i] + 1.5 * w
            if new < 0.0:
                new = 0.0
            dlam = new - lam[i]
            if dlam != 0.0:
                lam[i] = new
                res = res + E[:, i] * dlam
        iters = it + 1
        if float(np.max(np.abs(lam - lam_old))) < 1e-6:
            break
    n_u = mpc["N"] * mpc["nu"]
    U = -(Ginv @ (g + C.T @ lam))
    U = np.clip(U, np.full(n_u, -u_max), np.full(n_u, u_max))
    d_now = CGinv @ g + d
    stats = {"iter": iters, "n_constraints": m, "active": int(np.sum(np.abs(C @ U - d_now) < 1e-3))}
    return float(U[0]), lam, stats
