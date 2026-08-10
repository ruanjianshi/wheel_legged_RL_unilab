#!/usr/bin/env python3
"""Calibrate the xqrobotwl 2-link virtual-leg FK parameters for VMC control.

The VMC virtual leg is modelled as a planar 2-link chain in the sagittal plane:
    theta1 = s1 * q_hip + c1
    theta2 = s2 * q_knee + c2
    end_x  = offset + l1*cos(theta1) + l2*cos(theta1 + theta2)
    end_y  = l1*sin(theta1) + l2*sin(theta1 + theta2)
    L0     = sqrt(end_x**2 + end_y**2)
    theta0 = atan2(end_y, end_x) - pi/2

xqrobotwl joint axes differ from the reference robots, so the sign conventions
(s1, s2) and neutral offsets (c1, c2) are NOT assumed -- this script sweeps the
real MuJoCo model, reads the wheel position relative to the hip-pitch joint in
the base frame, and fits l1/l2/offset (linear) over a fine (s1,s2,c1,c2) grid.

IMPORTANT xqrobotwl geometry finding: at q_knee=0 the calf already points
forward-down while the thigh points backward-down (knee "neutral" is ~96deg of
bend), so the achievable leg length is roughly [0.15, 0.50] m -- the leg cannot
fully straighten within the joint range.

Usage:
    uv run scripts/calibrate_xqrobotwl_vmc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unilab.assets import ASSETS_ROOT_PATH  # noqa: E402
from unilab.base.backend import create_backend  # noqa: E402
from unilab.base.scene import SceneCfg  # noqa: E402

# Default posture from the scene keyframe (dof order):
# [L_roll, L_pitch, L_knee, R_roll, R_pitch, R_knee, L_wheel, R_wheel]
DEFAULT_LEG = np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15, 0.0, 0.0], dtype=np.float64)
# Sagittal joint indices in dof order
HIP_L, KNEE_L, HIP_R, KNEE_R = 1, 2, 4, 5
NQ, NV = 15, 14
# dof index -> position in qpos[7:] (qpos is interleaved: L_roll,L_pitch,L_knee,L_wheel,R_roll,R_pitch,R_knee,R_wheel)
DOF_TO_QPOS = np.array([0, 1, 2, 4, 5, 6, 3, 7], dtype=np.int64)


def build_backend():
    scene = SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotwl" / "scene_flat_vmc.xml")
    )
    backend = create_backend(
        "mujoco", scene, 1, 0.005, base_name="base_link", add_body_sensors=True
    )
    backend.materialize()
    return backend


def _base_qpos(hip: float, knee: float, side: str) -> np.ndarray:
    """Build a qpos with the given sagittal joint values; hip_roll zeroed for a clean planar fit."""
    qpos = np.zeros(NQ, dtype=np.float64)
    qpos[2] = 0.65  # base z
    qpos[3] = 1.0  # quat w
    qpos[7:] = DEFAULT_LEG
    qpos[7 + DOF_TO_QPOS[0]] = 0.0  # L_hip_roll -> 0 for a clean planar fit
    qpos[7 + DOF_TO_QPOS[3]] = 0.0  # R_hip_roll -> 0
    if side == "L":
        qpos[7 + DOF_TO_QPOS[HIP_L]] = hip
        qpos[7 + DOF_TO_QPOS[KNEE_L]] = knee
    else:
        qpos[7 + DOF_TO_QPOS[HIP_R]] = hip
        qpos[7 + DOF_TO_QPOS[KNEE_R]] = knee
    return qpos


def sweep_leg(
    backend, side: str, n: int = 11
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep the sagittal joints of one leg. Returns end_x, end_y, qhip, qknee."""
    base = "left" if side == "L" else "right"
    hip_ids = backend.get_body_ids([f"{base}_link_2", f"{base}_link_wheel"])
    if side == "L":
        qhips = np.linspace(-1.047, 2.094, n)
    else:
        qhips = np.linspace(-2.094, 1.047, n)
    qknees = np.linspace(-0.873, 0.873, n)
    end_x, end_y, qh_list, qk_list = [], [], [], []
    qvel = np.zeros(NV, dtype=np.float64)
    for qh in qhips:
        for qk in qknees:
            backend.set_state(np.array([0], dtype=np.int32), _base_qpos(qh, qk, side), qvel)
            pos = backend.get_body_pos_b(hip_ids)  # (1, 2, 3): [hipframe, wheel]
            v = pos[0, 1] - pos[0, 0]  # wheel - hip_pitch joint (base frame)
            end_x.append(float(v[0]))
            end_y.append(-float(v[2]))
            qh_list.append(float(qh))
            qk_list.append(float(qk))
    return np.asarray(end_x), np.asarray(end_y), np.asarray(qh_list), np.asarray(qk_list)


def _fit_linear(end_x, end_y, t1, t2) -> tuple[float, float, float, float]:
    """Linear least squares for (offset, l1, l2) given theta1/theta2. Returns (rmse, offset, l1, l2)."""
    N = end_x.shape[0]
    A = np.zeros((2 * N, 3), dtype=np.float64)
    A[:N, 0] = 1.0
    A[:N, 1] = np.cos(t1)
    A[:N, 2] = np.cos(t1 + t2)
    A[N:, 1] = np.sin(t1)
    A[N:, 2] = np.sin(t1 + t2)
    b = np.concatenate([end_x, end_y])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    rmse = float(np.sqrt(np.mean((A @ sol - b) ** 2)))
    return rmse, float(sol[0]), float(sol[1]), float(sol[2])


def fit_model(end_x, end_y, qhip, qknee):
    """Fit (s1,s2,c1,c2, offset,l1,l2) with coarse grid + local refinement."""
    best = None

    def evaluate(s1, s2, c1, c2):
        t1 = s1 * qhip + c1
        t2 = s2 * qknee + c2
        return _fit_linear(end_x, end_y, t1, t2)

    # Coarse global grid
    for s1 in (1.0, -1.0):
        for s2 in (1.0, -1.0):
            for c1 in np.linspace(-np.pi, np.pi, 40):
                for c2 in np.linspace(-np.pi, np.pi, 40):
                    rmse, offset, l1, l2 = evaluate(s1, s2, c1, c2)
                    if best is None or rmse < best[0]:
                        best = (rmse, s1, s2, c1, c2, offset, l1, l2)
    # Local refinement around the best (c1, c2)
    rmse, s1, s2, c1, c2, *_ = best
    step = np.pi / 16
    for _ in range(2):
        for dc1 in np.linspace(-step, step, 32):
            for dc2 in np.linspace(-step, step, 32):
                r, offset, l1, l2 = evaluate(s1, s2, c1 + dc1, c2 + dc2)
                if r < best[0]:
                    best = (r, s1, s2, c1 + dc1, c2 + dc2, offset, l1, l2)
    return best


def canonicalize(s1, s2, c1, c2, offset, l1, l2):
    """Force l1, l2 positive by folding sign flips into the phase offsets.

    Flipping l1 -> -l1 requires c1 += pi AND c2 -= pi so that the coupling
    theta1 + theta2 (and therefore the l2 term) stays invariant. Flipping l2
    -> -l2 requires c2 += pi only (theta1 stays fixed).
    """
    if l1 < 0:
        l1, c1 = -l1, c1 + np.pi
        c2 = c2 - np.pi
    if l2 < 0:
        l2, c2 = -l2, c2 + np.pi
    return s1, s2, c1, c2, offset, l1, l2


def _rmse(end_x, end_y, t1, t2, offset, l1, l2) -> float:
    px = offset + l1 * np.cos(t1) + l2 * np.cos(t1 + t2)
    py = l1 * np.sin(t1) + l2 * np.sin(t1 + t2)
    return float(np.sqrt(np.mean((px - end_x) ** 2 + (py - end_y) ** 2)))


def main() -> int:
    backend = build_backend()
    try:
        # ---- Fit each leg independently, then canonicalize to positive l1/l2 ----
        lex, ley, lqh, lqk = sweep_leg(backend, "L")
        rex, rey, rqh, rqk = sweep_leg(backend, "R")
        l_fit = fit_model(lex, ley, lqh, lqk)
        r_fit = fit_model(rex, rey, rqh, rqk)
        l_rmse, l_s1, l_s2, l_c1, l_c2, l_off, l_l1, l_l2 = (
            l_fit[0],
            *canonicalize(*l_fit[1:]),
        )
        r_rmse, r_s1, r_s2, r_c1, r_c2, r_off, r_l1, r_l2 = (
            r_fit[0],
            *canonicalize(*r_fit[1:]),
        )

        print("=== LEFT LEG FIT (canonicalized, positive l1/l2) ===")
        print(f"RMSE={l_rmse * 1000:.3f} mm  s1={l_s1:+g} s2={l_s2:+g} c1={l_c1:.4f} c2={l_c2:.4f}")
        print(f"l1={l_l1:.5f} l2={l_l2:.5f} offset={l_off:.5f}")
        print("=== RIGHT LEG FIT (canonicalized) ===")
        print(f"RMSE={r_rmse * 1000:.3f} mm  s1={r_s1:+g} s2={r_s2:+g} c1={r_c1:.4f} c2={r_c2:.4f}")
        print(f"l1={r_l1:.5f} l2={r_l2:.5f} offset={r_off:.5f}")

        # ---- Unified convention: per-leg signs, shared offsets / link lengths ----
        hip_sign = np.array([l_s1, r_s1])
        knee_sign = np.array([l_s2, r_s2])
        c1_shared = float(np.mean([l_c1, r_c1]))
        c2_shared = float(np.mean([l_c2, r_c2]))
        l1 = float(np.mean([l_l1, r_l1]))
        l2 = float(np.mean([l_l2, r_l2]))
        offset = float(np.mean([l_off, r_off]))

        # Verify unified convention against both legs
        for name, (ex, ey, qh, qk, i) in {
            "L": (lex, ley, lqh, lqk, 0),
            "R": (rex, rey, rqh, rqk, 1),
        }.items():
            t1 = hip_sign[i] * qh + c1_shared
            t2 = knee_sign[i] * qk + c2_shared
            rmse_u = _rmse(ex, ey, t1, t2, offset, l1, l2)
            print(f"UNIFIED {name}: RMSE={rmse_u * 1000:.2f} mm")

        # ---- Default posture -> offsets ----
        t1d = hip_sign[0] * DEFAULT_LEG[HIP_L] + c1_shared
        t2d = knee_sign[0] * DEFAULT_LEG[KNEE_L] + c2_shared
        dx = offset + l1 * np.cos(t1d) + l2 * np.cos(t1d + t2d)
        dy = l1 * np.sin(t1d) + l2 * np.sin(t1d + t2d)
        l0_default = float(np.sqrt(dx**2 + dy**2))
        theta0_default = float(np.arctan2(dy, dx) - np.pi / 2)

        # L0 range over the left leg joint grid
        t1g = hip_sign[0] * lqh + c1_shared
        t2g = knee_sign[0] * lqk + c2_shared
        l0_grid = np.sqrt(
            (offset + l1 * np.cos(t1g) + l2 * np.cos(t1g + t2g)) ** 2
            + (l1 * np.sin(t1g) + l2 * np.sin(t1g + t2g)) ** 2
        )
        print("\n=== DEFAULT POSTURE ===")
        print(f"l0_default = {l0_default:.4f} m")
        print(f"theta0_default = {theta0_default:.4f} rad")
        print(f"L0 range = [{l0_grid.min():.4f}, {l0_grid.max():.4f}] m")

        print("\n=== XqRobotWLVMCConfig (unified, ready to paste) ===")
        print(f"l1: {l1:.5f}")
        print(f"l2: {l2:.5f}")
        print(f"offset: {offset:.5f}")
        print(f"theta0_offset: {theta0_default:.4f}")
        print(f"l0_offset: {l0_default:.4f}")
        print(f"l0_min: {max(0.10, float(l0_grid.min()) + 0.03):.4f}")
        print(f"l0_max: {float(l0_grid.max()) - 0.03:.4f}")
        print(f"hip_sign: [{l_s1:+.1f}, {r_s1:+.1f}]")
        print(f"knee_sign: [{l_s2:+.1f}, {r_s2:+.1f}]")
        print(f"c1: {c1_shared:.4f}")
        print(f"c2: {c2_shared:.4f}")
        print("singularity_epsilon: 0.05")
        return 0
    finally:
        backend.cleanup_scene_assets()


if __name__ == "__main__":
    raise SystemExit(main())
