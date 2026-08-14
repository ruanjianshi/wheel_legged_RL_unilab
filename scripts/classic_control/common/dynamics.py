"""矢状面倒立摆动力学 — 离线数值线性化 (复刻 single_leg_lqr.meas_pitch_alpha_beta).

模型: θ̈ = α·θ + β·u  (θ 前倾角, u 轮线加速度 m/s²)
状态 x = [θ, θ̇, v, xpos], 控制 u = 轮线加速度。

测量方式: raw MuJoCo (scene_flat.xml), 直立站姿下 mj_step 长窗二阶拟合:
  - α: θ₀=0.05 rad 扰动无控制, 0.02s 后 α = 2(θ_n−θ₀)/(win²·θ₀)
  - β: θ=0, 恒定 u=1 m/s² (轮速积分), β = 2(θ_n−θ₀)/(win²·u)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 mujoco") from e

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.classic_control.common.config import (
    STANDING_ANGLES,
    STANDING_BASE_Z,
    VMC,
    WHEEL_R,
)

FLAT_XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"


def standing_qpos() -> np.ndarray:
    """直立站姿 qpos (15D): [x,y,z,quat(wxyz), 8关节]."""
    qpos = np.zeros(15, dtype=np.float64)
    qpos[2] = STANDING_BASE_Z
    qpos[3] = 1.0  # 直立四元数
    qpos[7:15] = [*STANDING_ANGLES[:3], 0.0, *STANDING_ANGLES[3:], 0.0]
    return qpos


def _theta(d: mujoco.MjData, adr_up: int) -> float:
    """前倾角 (绕 base y 轴, 前倾为正): up 向量 x/z."""
    up = d.sensordata[adr_up : adr_up + 3]
    return float(np.arctan2(up[0], up[2]))


def _hold(d: mujoco.MjData, wheel_vel: float) -> None:
    """姿态锁: 腿 position 伺服钉 standing_angles, 轮速度 w."""
    # actuator 顺序 [L_roll,L_pitch,L_knee,L_wheel,R_roll,R_pitch,R_knee,R_wheel]
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[0:3] = STANDING_ANGLES[:3]
    ctrl[3] = wheel_vel
    ctrl[4:7] = STANDING_ANGLES[3:]
    ctrl[7] = wheel_vel
    d.ctrl[:] = ctrl


def measure_alpha_beta(
    model: mujoco.MjModel, dt: float = 0.01, window: float = 0.03
) -> tuple[float, float]:
    """测量 (α, β): θ̈ = α·θ + β·u.

    α: θ₀=0.05 扰动, 每步钳位轮 qvel=0 (固定 cart → 纯倒立摆重力失稳).
    β: θ=0, 恒定 u=1 m/s² (轮速积分), 测耦合.
    """
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    n = max(int(window / dt), 2)
    # qvel 轮索引 (freejoint 6 + [L_roll,L_pitch,L_knee,L_wheel,R_roll,R_pitch,R_knee,R_wheel])
    WHEEL_QVEL = (9, 13)

    # α: θ₀=0.05 扰动, 无控制, 轮钳位
    d = mujoco.MjData(model)
    d.qpos[:] = standing_qpos()
    # 绕 base 局部 y 轴转 +0.05 rad (前倾扰动)
    qw, qx, qy, qz = d.qpos[3:7]
    half = 0.025
    d.qpos[3:7] = [
        qw * np.cos(half) - qy * np.sin(half),
        qx,
        qy * np.cos(half) + qw * np.sin(half),
        qz,
    ]
    mujoco.mj_forward(model, d)
    th0 = _theta(d, adr_up)
    for _ in range(n):
        _hold(d, 0.0)
        mujoco.mj_step(model, d)
        d.qvel[list(WHEEL_QVEL)] = 0.0  # 钳位轮 (cart 固定)
    thn = _theta(d, adr_up)
    alpha = 2.0 * (thn - th0) / (window * window * th0)

    # β: θ=0, 恒定 u=1 m/s² (轮速线性积分)
    d2 = mujoco.MjData(model)
    d2.qpos[:] = standing_qpos()
    mujoco.mj_forward(model, d2)
    thb0 = _theta(d2, adr_up)
    a = 1.0
    w = 0.0
    for _ in range(n):
        w += (a / WHEEL_R) * dt
        _hold(d2, w)
        mujoco.mj_step(model, d2)
    thb1 = _theta(d2, adr_up)
    beta = 2.0 * (thb1 - thb0) / (window * window * a)

    return float(alpha), float(beta)


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


def build_discrete(
    alpha: float, beta: float, dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造连续 A_c/B_c → ZOH 离散 A_d/B_d. 返回 (A_c, B_c, A_d, B_d)."""
    A_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [alpha, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    B_c = np.array([[0.0], [beta], [1.0], [0.0]], dtype=np.float64)

    expm = _expm

    nx = 4
    M = np.zeros((nx + 1, nx + 1), dtype=np.float64)
    M[:nx, :nx] = A_c
    M[:nx, nx:] = B_c
    expM = expm(M * dt)
    A_d = expM[:nx, :nx]
    B_d = expM[:nx, nx:]
    return A_c, B_c, A_d, B_d


def analytic(alpha: float, beta: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """解析倒立摆模型 (cart-pole): θ̈=αθ+βu. 返回 (A_d, B_d)."""
    _, _, A_d, B_d = build_discrete(alpha, beta, dt)
    return A_d, B_d


def discretize_zoh(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """连续 A_c/B_c → ZOH 离散 A_d/B_d (指数积分, 复用 _expm)."""
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


def measure_and_save(dt: float = 0.01, out_path: str | Path | None = None) -> dict:
    """测量 α,β → A_d/B_d, 可选存 npz. 返回 dict."""
    model = mujoco.MjModel.from_xml_path(str(FLAT_XML))
    model.opt.timestep = dt
    alpha, beta = measure_alpha_beta(model, dt=dt)
    A_c, B_c, A_d, B_d = build_discrete(alpha, beta, dt)
    result = {
        "alpha": alpha,
        "beta": beta,
        "A_d": A_d,
        "B_d": B_d,
        "dt": dt,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            **{k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in result.items()},
        )
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "logs" / "classic" / "dynamics_flat.npz"))
    ap.add_argument("--dt", type=float, default=0.01)
    args = ap.parse_args()
    r = measure_and_save(dt=args.dt, out_path=args.out)
    print(f"α = {r['alpha']:.4f}  β = {r['beta']:.4f}  dt = {r['dt']}")
    print("A_d =\n", np.array2string(r["A_d"], precision=4))
    print("B_d =\n", np.array2string(r["B_d"], precision=4))
    print(f"saved → {args.out}")
