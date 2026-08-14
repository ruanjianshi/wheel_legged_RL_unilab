"""MPC×SAC 融合分支 — 矢状面倒立摆动力学 (轮速命令模型).

移植自经典轨 scripts/classic_control/common/dynamics.py 的 velocity_command_model
(逐行一致, 独立拥有). 模型: θ̈=αθ+β·v̇, v̇=(−uR−v)/τ (控制器轮映射 ctrl=−u).
状态 x=[θ,θ̇,v,xpos], 控制 u=轮速命令 (rad/s).
"""

from __future__ import annotations

import numpy as np

from scripts.classic_control.common.config import WHEEL_R


def _expm(A: np.ndarray) -> np.ndarray:
    """矩阵指数 (Taylor + scaling-and-squaring, 无 scipy 依赖). 小矩阵够用."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    norm = float(np.linalg.norm(A, ord=np.inf))
    s = int(np.ceil(np.log2(norm))) if norm > 1.0 else 0
    B = A / (2.0**s)
    R = np.eye(n) + B
    term = B.copy()
    for k in range(2, 16):
        term = term @ B / k
        R = R + term
    for _ in range(s):
        R = R @ R
    return R


def discretize_zoh(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """连续 A_c/B_c → ZOH 离散 A_d/B_d (指数积分)."""
    nx = A_c.shape[0]
    M = np.zeros((nx + 1, nx + 1), dtype=np.float64)
    M[:nx, :nx] = A_c
    M[:nx, nx:] = B_c
    expM = _expm(M * dt)
    return expM[:nx, :nx], expM[:nx, nx:]


def velocity_command_model(
    alpha: float, beta: float, tau: float, dt: float = 0.01, wheel_r: float = WHEEL_R
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """轮速命令模型 (w_c convention, rad/s): v̇=(u·R−v)/τ, θ̈=αθ+β·v̇.

    状态 x=[θ,θ̇,v,xpos], 控制 u=轮速命令 (rad/s)。返回 (A_c, B_c, A_d, B_d)。
    - v 为轮线速度 (qvel_L·R, 前向为正); θ̈ 由轮加速 v̇ 驱动 (β<0 车摆耦合)。
    - 符号: u>0 → v̇<0 (后驱) → θ̈=β·v̇>0 (前倾增大), 与实测方向一致。
    """
    A_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [alpha, 0.0, -beta / tau, 0.0],
            [0.0, 0.0, -1.0 / tau, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    # ★ 控制器轮映射 ctrl=−u → ω̇=(ctrl−ω)/τ ⇒ v̇=ω̇R = −(R/τ)u − (1/τ)v
    #   (B_c[2] 为负: u>0 → v̇<0 后驱, 与实测 v→−uR 一致)
    B_c = np.array([[0.0], [-beta * wheel_r / tau], [-wheel_r / tau], [0.0]], dtype=np.float64)
    A_d, B_d = discretize_zoh(A_c, B_c, dt)
    return A_c, B_c, A_d, B_d


def load_plant_model(
    alpha: float,
    beta: float,
    tau: float,
    dt: float = 0.01,
    model_file: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """加载离散模型: ① npz 黑箱 (A_d/B_d) ② 解析模型. 返回 (A_d, B_d)."""
    from pathlib import Path

    if model_file:
        p = Path(model_file)
        if not p.is_absolute():
            # 相对路径相对项目根 (logs/...) — 本模块在 scripts/fusion_control/mpc_sac/mpc/ (深 4 层)
            root = Path(__file__).resolve().parents[4]
            p = root / p
        if p.exists():
            _d = np.load(p)
            return np.asarray(_d["A_d"], dtype=np.float64), np.asarray(_d["B_d"], dtype=np.float64).reshape(-1, 1)
    return velocity_command_model(alpha, beta, tau, dt, WHEEL_R)[2:]
