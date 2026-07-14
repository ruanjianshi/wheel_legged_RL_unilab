"""XqRobotV2 model visualization using Pinocchio + CJ-003 URDF.

Loads the CJ-003 URDF (identical 8DOF structure to XqRobotV2), computes FK
for toe-walk reference trajectories, and visualizes joints/wheels in 3D.

Usage:
    cd /home/robot/xiaoq/wheel_legged_RL_unilab
    python tools/pinocchio_traj/load_model_viz.py
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

# ── Load model ──
URDF_PATH = (
    "/home/robot/xiaoq/projects/wheel_legged_genesis/assets/urdf/CJ-003/urdf/CJ-003-wheelfoot.urdf"
)

model = pin.buildModelFromUrdf(URDF_PATH)
data = model.createData()

print(f"Model: {model.name}")
print(f"  Joints: {model.njoints}")
print(f"  Bodies: {model.nframes}")

# Print joint table
print("\nJoint layout:")
for i in range(1, model.njoints):  # skip universe joint (0)
    name = model.names[i]
    jid = model.joints[i]
    parent = model.parents[i]
    nq = model.nqs[i]
    lower = model.lowerPositionLimit[i]
    upper = model.upperPositionLimit[i]
    print(f"  [{i}] {name:30s} parent={parent} nq={nq} limit=[{lower:.2f}, {upper:.2f}]")

# ── CJ-003 → XqRobotV2 joint name mapping ──
# CJ-003 names: left_hip_joint(0), left_thigh_joint(1), left_calf_joint(2),
#                right_hip_joint(3), right_thigh_joint(4), right_calf_joint(5),
#                left_wheel_joint(6), right_wheel_joint(7)

# Build joint index map
joint_names = [model.names[i] for i in range(1, model.njoints)]
print("\nActuated joints:", joint_names[:8])


# ── Toe-walk reference trajectory ──
def generate_toe_walk_ref(t, cycle_time=0.5, ref_scale=0.12):
    """Generate reference joint angles (8D) for toe-walking."""
    phase = t / cycle_time
    sin_pos = np.sin(2 * np.pi * phase)
    cos_pos = np.cos(2 * np.pi * phase)

    # Defaults: all joints at 0 (CJ-003 convention)
    ref = np.zeros(8)

    left_calf_active = np.clip(-cos_pos, 0, 1)
    right_calf_active = np.clip(cos_pos, 0, 1)
    left_thigh_active = np.clip(-sin_pos, 0, 1)
    right_thigh_active = np.clip(sin_pos, 0, 1)

    # Thigh: slight swing
    ref[1] = left_thigh_active * ref_scale * 0.5
    ref[4] = right_thigh_active * ref_scale * 0.5
    # Calf: big bend for lift
    ref[2] = -left_calf_active * ref_scale * 5
    ref[5] = -right_calf_active * ref_scale * 5
    # Wheels stay at 0
    ref[6] = 0.0
    ref[7] = 0.0

    return ref


# ── Forward kinematics for the full trajectory ──
num_steps = 80
t_vals = np.linspace(0, 0.5, num_steps)

# Find wheel bodies (CJ-003: left_wheel_joint, right_wheel_joint → last 2 bodies)
wheel_frame_names = []
for i in range(model.nframes):
    name = model.frames[i].name
    if "wheel" in name.lower() or "foot" in name.lower():
        wheel_frame_names.append((i, name))
print("\nWheel/foot frames:", wheel_frame_names)

# Get base joint and wheel joint IDs (joint 1 = base, joints 7,8 = wheels approximate)
# Actually, let me just compute FK at each joint and track Z positions
wheel_z_l = np.zeros(num_steps)
wheel_z_r = np.zeros(num_steps)

for idx, t in enumerate(t_vals):
    ref = generate_toe_walk_ref(t)
    q = np.zeros(model.nq)
    # Set base position: x=0, y=0, z=0.65
    q[0] = 0.0
    q[1] = 0.0
    q[2] = 0.65
    q[3:7] = [1, 0, 0, 0]  # identity quat

    # Set joint angles: model uses cos/sin repr (nq=2 per continuous joint)
    q[model.nqs[0] :] = 0.0  # reset joint space
    idx = model.nqs[0]  # start after freejoint
    for j, angle in enumerate(ref):
        c = np.cos(angle)
        s = np.sin(angle)
        q[idx] = c
        q[idx + 1] = s
        idx += 2  # each continuous joint takes 2 DOFs

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Get wheel positions from frames (use "Link" frames, not "motor" or "joint")
    for fid, fname in wheel_frame_names:
        pos = data.oMf[fid].translation
        if "left" in fname.lower() and "Link" in fname:
            wheel_z_l[idx] = pos[2]
        elif "right" in fname.lower() and "Link" in fname:
            wheel_z_r[idx] = pos[2]

# ── Plot ──
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

ax = axes[0]
ax.plot(t_vals, wheel_z_l, "b-", label="Left wheel Z", linewidth=2)
ax.plot(t_vals, wheel_z_r, "r--", label="Right wheel Z", linewidth=2)
ax.axhline(y=0.11, color="gray", linestyle=":", label="Ground (r=0.11m)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Wheel Z (m)")
ax.set_title("XqRobotV2 Toe-Walk: Wheel Height (Pinocchio FK)")
ax.legend()
ax.grid(True)

# Ground clearance
ax = axes[1]
# Ground clearance (relative to default Z)
default_l = wheel_z_l[0]  # Z at t=0 (default stance)
default_r = wheel_z_r[0]
clearance_l = wheel_z_l - default_l
clearance_r = wheel_z_r - default_r
ax.plot(t_vals, clearance_l, "b-", linewidth=2)
ax.plot(t_vals, clearance_r, "r--", linewidth=2)
ax.axhline(y=0, color="gray", linestyle=":")
ax.fill_between(t_vals, 0, np.maximum(clearance_l, 0), alpha=0.2, color="b")
ax.fill_between(t_vals, 0, np.maximum(clearance_r, 0), alpha=0.2, color="r")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Clearance (m)")
ax.set_title("Wheel Ground Clearance")
ax.legend()
ax.grid(True)

plt.suptitle("Toe-Walk Reference Trajectory (1 cycle, 0.5s)", fontsize=14)
plt.tight_layout()
plt.savefig(
    "/home/robot/xiaoq/wheel_legged_RL_unilab/tools/pinocchio_traj/toe_walk_pinocchio.png", dpi=150
)
plt.close()

print(f"\nMax lift: L={np.max(clearance_l) * 1000:.0f}mm, R={np.max(clearance_r) * 1000:.0f}mm")
print("Saved: tools/pinocchio_traj/toe_walk_pinocchio.png")
