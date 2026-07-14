"""Generate GIF animation of XqRobotV2 toe-walk reference trajectory.

Uses Pinocchio FK with XqRobotV2's own URDF.

Usage:
    conda activate unilab
    cd /home/robot/xiaoq/wheel_legged_RL_unilab
    python tools/pinocchio_traj/animate_toe_walk.py

Output: tools/pinocchio_traj/xqrobotV2_toe_walk.gif
"""

from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.expanduser("~/miniconda3/envs/unilab/lib/python3.10/site-packages"))
import pinocchio as pin

# ── Load XqRobotV2 URDF ──
URDF_DIR = "/home/robot/xiaoq/wheel_legged_RL_unilab/tools/xqrobotV2"
os.chdir(URDF_DIR)

model = pin.buildModelFromUrdf("urdf/xqrobotV2.urdf")
data = model.createData()

print(f"Model: {model.name}, Joints: {model.njoints}")
for i in range(1, model.njoints):
    name = model.names[i]
    limits = (model.lowerPositionLimit[i], model.upperPositionLimit[i])
    print(f"  [{i}] {name:25s} limit=[{limits[0]:.2f}, {limits[1]:.2f}]")


def generate_toe_walk_ref(t, cycle_time=0.5, ref_scale=0.12):
    phase = t / cycle_time
    sin_pos = np.sin(2 * np.pi * phase)
    cos_pos = np.cos(2 * np.pi * phase)
    ref = np.zeros(8)
    ref[0] = 0.1
    ref[3] = 0.1
    ref[1] = 0.1 + np.clip(-sin_pos, 0, 1) * ref_scale * 0.5
    ref[4] = 0.1 + np.clip(sin_pos, 0, 1) * ref_scale * 0.5
    ref[2] = -0.1 - np.clip(-cos_pos, 0, 1) * ref_scale * 5
    ref[5] = -0.1 - np.clip(cos_pos, 0, 1) * ref_scale * 5
    return ref


def set_q_from_angles(angles):
    """Build q vector from 8 joint angles. Handles mixed nq=1 (revolute) and nq=2 (continuous) joints."""
    q = np.zeros(model.nq)
    qi = 0  # position in q array
    ai = 0  # position in angles array
    for i in range(1, model.njoints):  # skip universe (0)
        nq = model.nqs[i]
        if nq == 1:
            q[qi] = angles[ai]
            qi += 1
            ai += 1
        elif nq == 2:
            q[qi] = np.cos(angles[ai])
            q[qi + 1] = np.sin(angles[ai])
            qi += 2
            ai += 1
        else:
            qi += nq
    return q


# ── Compute trajectory ──
num_frames = 80
dt = 0.00625
lw_z = np.zeros(num_frames)
rw_z = np.zeros(num_frames)
for i in range(num_frames):
    angles = generate_toe_walk_ref(i * dt)
    lw_z[i], rw_z[i] = get_wheel_z(angles)

clearance_l = lw_z - lw_z[0]
clearance_r = rw_z - rw_z[0]
print(f"Max lift: L={np.max(clearance_l) * 1000:.0f}mm, R={np.max(clearance_r) * 1000:.0f}mm")

# ── Animation ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
times = np.linspace(0, (num_frames - 1) * dt, num_frames)

for frame in range(0, num_frames, 5):
    fig.clf()
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    t = frame * dt

    # 3D stick figure
    ax.plot([0, 0], [-0.16, 0.16], [0.65, 0.65], "k-", linewidth=6)
    lx, rx = 0, 0
    ly = -0.16 + (clearance_l[frame] > 0) * 0.03
    ry = 0.16 - (clearance_r[frame] > 0) * 0.03
    lz = 0.65 - 0.55 + lw_z[frame] * 0.8
    rz = 0.65 - 0.55 + rw_z[frame] * 0.8
    ax.plot([0, lx], [0, ly], [0.65, lz], "b-", linewidth=3)
    ax.plot([0, rx], [0, ry], [0.65, rz], "r-", linewidth=3)
    ax.scatter(lx, ly, lz, color="b", s=80)
    ax.scatter(rx, ry, rz, color="r", s=80)
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.8)
    ax.set_title(f"3D (t={t:.2f}s)")
    ax.view_init(elev=20, azim=45)

    # Wheel height
    ax2.plot(times, lw_z, "b-", alpha=0.5)
    ax2.plot(times, rw_z, "r--", alpha=0.5)
    ax2.axvline(x=t, color="gray", alpha=0.5)
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Wheel Height")
    ax2.grid(True, alpha=0.3)

    # Clearance
    ax3.fill_between(times, 0, np.maximum(clearance_l, 0) * 1000, alpha=0.2, color="b")
    ax3.fill_between(times, 0, np.maximum(clearance_r, 0) * 1000, alpha=0.2, color="r")
    ax3.plot(times, clearance_l * 1000, "b-", linewidth=1.5)
    ax3.plot(times, clearance_r * 1000, "r--", linewidth=1.5)
    ax3.axvline(x=t, color="gray", alpha=0.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("mm")
    ax3.set_title("Clearance")
    ax3.grid(True, alpha=0.3)

    # Joint angles
    angles = generate_toe_walk_ref(t)
    ax4.barh(
        ["L hip", "L thigh", "L calf", "L whl", "R hip", "R thigh", "R calf", "R whl"],
        angles,
        color=["b"] * 4 + ["r"] * 4,
        alpha=0.7,
    )
    ax4.set_xlim(-0.6, 0.6)
    ax4.set_title(f"Joint Angles (t={t:.2f}s)")
    ax4.axvline(x=0, color="gray", alpha=0.3)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("XqRobotV2 Toe-Walk Reference Trajectory", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"/tmp/toe_frame_{frame:03d}.png", dpi=80)
    plt.close()

# Combine into GIF
from PIL import Image

frames_gif = []
for i in range(0, num_frames, 2):
    img = Image.open(f"/tmp/toe_frame_{i * 5:03d}.png")
    frames_gif.append(img)

output = "/home/robot/xiaoq/wheel_legged_RL_unilab/tools/pinocchio_traj/xqrobotV2_toe_walk.gif"
frames_gif[0].save(output, save_all=True, append_images=frames_gif[1:], duration=100, loop=0)
for i in range(0, num_frames, 5):
    os.remove(f"/tmp/toe_frame_{i:03d}.png")
print(f"\nSaved: {output} ({os.path.getsize(output) / 1024 / 1024:.1f}MB)")
