"""XqRobotV2 joint direction calibration tool.

Press 1-8 to select a joint, then ↑↓ to rotate it.
Selected joint is highlighted in terminal output.

Usage:
    conda activate unilab
    cd /home/robot/xiaoq/wheel_legged_RL_unilab
    python tools/mujoco/calibrate_joints.py
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

SCENE_PATH = os.path.expanduser(
    "~/xiaoq/wheel_legged_RL_unilab/src/unilab/assets/robots/xqrobotV2/scene_flat.xml"
)

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

# Disable gravity for clear observation
model.opt.gravity[:] = 0.0

# Joint names and their qpos indices
JOINT_INFO = [
    (7, "left_joint_1", "L_hip"),
    (8, "left_joint_2", "L_thigh"),
    (9, "left_joint_3", "L_calf"),
    (10, "left_joint_wheel", "L_wheel"),
    (11, "right_joint_1", "R_hip"),
    (12, "right_joint_2", "R_thigh"),
    (13, "right_joint_3", "R_calf"),
    (14, "right_joint_wheel", "R_wheel"),
]

selected = 0  # 0-7
step = 0.1  # radians per keypress
gravity = False

print("=" * 60)
print("XqRobotV2 JOINT DIRECTION CALIBRATION")
print("=" * 60)
print("Keys: 1-8 select joint, ↑↓ rotate, G toggle gravity, R reset, Q quit")
print()
for i, (_, _, name) in enumerate(JOINT_INFO):
    print(f"  {i + 1}: {name}")
print()


def key_callback(keycode):
    global selected, gravity, step

    if ord("1") <= keycode <= ord("8"):
        selected = keycode - ord("1")
        qi, xml_name, short = JOINT_INFO[selected]
        val = data.qpos[qi]
        print(
            f"\n[{selected + 1}] {short} ({xml_name}) qpos_idx={qi}  angle={np.rad2deg(val):+.1f}°"
        )

    elif keycode == 265:  # ↑
        qi, _, short = JOINT_INFO[selected]
        data.qpos[qi] += step
        mujoco.mj_forward(model, data)
        val = data.qpos[qi]
        print(f"  {short} +{step:.1f} = {np.rad2deg(val):+.1f}°")

    elif keycode == 264:  # ↓
        qi, _, short = JOINT_INFO[selected]
        data.qpos[qi] -= step
        mujoco.mj_forward(model, data)
        val = data.qpos[qi]
        print(f"  {short} -{step:.1f} = {np.rad2deg(val):+.1f}°")

    elif keycode == ord("R"):
        data.qpos[7:15] = [0.1, 0.1, -0.1, 0, 0.0, 0.1, -0.1, 0]
        mujoco.mj_forward(model, data)
        print("\n[RESET] all joints to default")

    elif keycode == ord("G"):
        gravity = not gravity
        model.opt.gravity[:] = [0, 0, -9.81] if gravity else [0, 0, 0]
        print(f"\nGravity: {'ON' if gravity else 'OFF'}")

    elif keycode == ord("Q"):
        print("\nQuit: close MuJoCo window or press Esc")


# Reset to default pose
data.qpos[:7] = [0, 0, 0.65, 1, 0, 0, 0]
data.qpos[7:15] = [0.1, 0.1, -0.1, 0, 0.0, 0.1, -0.1, 0]
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)
