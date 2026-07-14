"""XqRobotV2 toe-walk reference trajectory visualization using Pinocchio.

Generates sinusoidal reference trajectories and visualizes the resulting
wheel positions in 3D to verify leg lifting during swing phase.

Usage:
    cd /home/robot/xiaoq/wheel_legged_RL_unilab
    python tools/pinocchio_traj/visualize_toe_walk.py
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

# ── XqRobotV2 kinematic parameters (from xqrobotV2.xml) ──

# Joint layout: [left_hip_roll, left_thigh_pitch, left_calf_pitch, left_wheel,
#                 right_hip_roll, right_thigh_pitch, right_calf_pitch, right_wheel]
NUM_JOINTS = 8
LEG_JOINTS = 6  # first 6 are leg joints

# Default angles
DEFAULT_ANGLES = np.array([0.1, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, 0.0])

# Link lengths (meters)
THIGH_LEN = 0.3  # thigh link
CALF_LEN = 0.3  # calf link
WHEEL_RADIUS = 0.11  # wheel radius for ground contact check

# Body origin offsets (estimated from XML)
HIP_BASE_OFFSET = np.array([0.069, -0.124, -0.001])  # left hip offset from base
THIGH_OFFSET = np.array([-0.070, -0.019, 0.000])  # thigh from hip
CALF_OFFSET = np.array([-0.224, 0.015, -0.200])  # calf from thigh
WHEEL_OFFSET = np.array([0.224, -0.037, -0.199])  # wheel from calf


# ── Simple FK (no Pinocchio model needed for basic visualization) ──


def fk_left_leg(angles: np.ndarray, base_pos: np.ndarray = np.zeros(3)):
    """Forward kinematics for left leg chain."""
    hip_roll, thigh, calf = angles[0], angles[1], angles[2]

    # Base → hip (in base frame, then rotate by hip roll around X)
    hip_pos = base_pos + HIP_BASE_OFFSET

    # Hip → thigh (rotate around hip X by hip_roll, then around Y by thigh)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(hip_roll), -np.sin(hip_roll)],
            [0, np.sin(hip_roll), np.cos(hip_roll)],
        ]
    )
    Ry_thigh = np.array(
        [[np.cos(thigh), 0, np.sin(thigh)], [0, 1, 0], [-np.sin(thigh), 0, np.cos(thigh)]]
    )
    R_thigh = Rx @ Ry_thigh
    knee_pos = hip_pos + R_thigh @ THIGH_OFFSET

    # Thigh → calf
    Ry_calf = np.array(
        [[np.cos(calf), 0, np.sin(calf)], [0, 1, 0], [-np.sin(calf), 0, np.cos(calf)]]
    )
    R_calf = R_thigh @ Ry_calf
    wheel_pos = knee_pos + R_calf @ CALF_OFFSET

    return hip_pos, knee_pos, wheel_pos


def fk_right_leg(angles: np.ndarray, base_pos: np.ndarray = np.zeros(3)):
    """Forward kinematics for right leg chain (mirrored Y offsets)."""
    hip_roll, thigh, calf = angles[3], angles[4], angles[5]

    right_hip_offset = HIP_BASE_OFFSET * np.array([1, -1, 1])
    right_thigh_offset = THIGH_OFFSET * np.array([1, -1, 1])
    right_calf_offset = CALF_OFFSET * np.array([1, -1, 1])
    right_wheel_offset = WHEEL_OFFSET * np.array([1, -1, 1])

    hip_pos = base_pos + right_hip_offset

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(hip_roll), -np.sin(hip_roll)],
            [0, np.sin(hip_roll), np.cos(hip_roll)],
        ]
    )
    Ry_thigh = np.array(
        [[np.cos(thigh), 0, np.sin(thigh)], [0, 1, 0], [-np.sin(thigh), 0, np.cos(thigh)]]
    )
    R_thigh = Rx @ Ry_thigh
    knee_pos = hip_pos + R_thigh @ right_thigh_offset

    Ry_calf = np.array(
        [[np.cos(calf), 0, np.sin(calf)], [0, 1, 0], [-np.sin(calf), 0, np.cos(calf)]]
    )
    R_calf = R_thigh @ Ry_calf
    wheel_pos = knee_pos + R_calf @ right_calf_offset

    return hip_pos, knee_pos, wheel_pos


# ── Sinusoidal reference trajectory generator (toe-walk) ──


def generate_toe_walk_ref(t: np.ndarray, cycle_time=0.5, ref_scale=0.12):
    """Generate reference joint angles for toe-walking at time t.

    Phase convention:
      sin < 0 → left leg active (swing), right leg stance
      sin >= 0 → right leg active (swing), left leg stance

    Hip roll stays at default.
    cos leads sin by 90°: calf bends (lift) before thigh swings (step).
    """
    phase = t / cycle_time
    sin_pos = np.sin(2 * np.pi * phase)
    cos_pos = np.cos(2 * np.pi * phase)

    ref = DEFAULT_ANGLES.copy()

    # cos < 0 → left calf active (bend first = lift)
    left_calf_active = np.clip(-cos_pos, 0, 1)
    right_calf_active = np.clip(cos_pos, 0, 1)
    # sin < 0 → left thigh active (swing forward = step)
    left_thigh_active = np.clip(-sin_pos, 0, 1)
    right_thigh_active = np.clip(sin_pos, 0, 1)

    # Thigh: slight forward swing (small from default 0.1)
    ref[1] += left_thigh_active * ref_scale * 0.5
    ref[4] += right_thigh_active * ref_scale * 0.5
    # Calf: big bend for lift (more negative from default -0.1)
    ref[2] -= left_calf_active * ref_scale * 5
    ref[5] -= right_calf_active * ref_scale * 5

    return ref


# ── Main visualization ──


def main():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Time steps over one full cycle
    num_steps = 80
    dt = 0.01  # 100Hz
    t = np.linspace(0, 0.5, num_steps)

    # Store wheel positions
    lw_z = np.zeros(num_steps)
    rw_z = np.zeros(num_steps)
    lw_y = np.zeros(num_steps)
    rw_y = np.zeros(num_steps)
    lw_x = np.zeros(num_steps)
    rw_x = np.zeros(num_steps)

    base_pos = np.array([0.0, 0.0, 0.65])

    for i, ti in enumerate(t):
        angles = generate_toe_walk_ref(ti)
        _, _, lwp = fk_left_leg(angles, base_pos)
        _, _, rwp = fk_right_leg(angles, base_pos)
        lw_x[i], lw_y[i], lw_z[i] = lwp
        rw_x[i], rw_y[i], rw_z[i] = rwp

    ground_clearance_l = lw_z - WHEEL_RADIUS
    ground_clearance_r = rw_z - WHEEL_RADIUS

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Wheel Z (height)
    ax = axes[0, 0]
    ax.plot(t, lw_z, "b-", label="Left wheel Z")
    ax.plot(t, rw_z, "r--", label="Right wheel Z")
    ax.axhline(y=WHEEL_RADIUS, color="gray", linestyle=":", label="Ground")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheel Z (m)")
    ax.set_title("Wheel Height (Z)")
    ax.legend()
    ax.grid(True)

    # Ground clearance
    ax = axes[0, 1]
    ax.plot(t, ground_clearance_l, "b-", label="Left clearance")
    ax.plot(t, ground_clearance_r, "r--", label="Right clearance")
    ax.axhline(y=0, color="gray", linestyle=":")
    ax.fill_between(t, 0, ground_clearance_l, alpha=0.2, color="b")
    ax.fill_between(t, 0, ground_clearance_r, alpha=0.2, color="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Clearance (m)")
    ax.set_title("Wheel Ground Clearance")
    ax.legend()
    ax.grid(True)

    # Wheel Y (lateral spread)
    ax = axes[1, 0]
    ax.plot(t, lw_y, "b-", label="Left wheel Y")
    ax.plot(t, rw_y, "r--", label="Right wheel Y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wheel Y (m)")
    ax.set_title("Wheel Lateral Position (Y)")
    ax.legend()
    ax.grid(True)

    # 3D trajectory (top-down: X vs Y)
    ax = axes[1, 1]
    ax.plot(lw_x, lw_y, "b-", label="Left wheel", alpha=0.7)
    ax.plot(rw_x, rw_y, "r--", label="Right wheel", alpha=0.7)
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (lateral)")
    ax.set_title("Wheel Trajectory (Top View)")
    ax.legend()
    ax.axis("equal")
    ax.grid(True)

    plt.suptitle("XqRobotV2 Toe-Walk Reference Trajectory (1 cycle, 0.5s)", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        "/home/robot/xiaoq/wheel_legged_RL_unilab/tools/pinocchio_traj/toe_walk_ref.png", dpi=150
    )
    plt.close()

    print(f"Max left wheel lift:  {np.max(ground_clearance_l) * 1000:.1f} mm")
    print(f"Max right wheel lift: {np.max(ground_clearance_r) * 1000:.1f} mm")
    print("Saved: tools/pinocchio_traj/toe_walk_ref.png")


if __name__ == "__main__":
    main()
