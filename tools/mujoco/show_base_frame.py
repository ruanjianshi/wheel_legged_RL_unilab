"""Display XqRobotV2 base_link coordinate axes in MuJoCo viewer.

Coordinate convention (verified against joint offsets in XML):
  X = Red  (forward, +X → robot front)
  Y = Green (right, +Y → robot right)
  Z = Blue  (up, +Z → sky)

Evidence: left_hip at pos.y=-0.124, right_hip at pos.y=+0.124 → Y=right.

Usage:
    uv run tools/mujoco/show_base_frame.py
"""

import os
import time

import mujoco
import mujoco.viewer
import numpy as np

SCENE_PATH = os.path.expanduser(
    "~/xiaoq/wheel_legged_RL_unilab/src/unilab/assets/robots/xqrobotV2/scene_flat.xml"
)

AXIS_LENGTH = 0.35
AXIS_RADIUS = 0.010

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

model.opt.gravity[:] = [0, 0, -9.81]

key_id = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

body_id = model.body("base_link").id

print("=" * 50)
print("XqRobotV2 BASE COORDINATE FRAME")
print(f"Base pos:  {data.xpos[body_id]}")
print(f"Base quat: {data.xquat[body_id]}")
print(f"Left hip offset:  {data.xpos[model.body('left_link_1').id] - data.xpos[body_id]}")
print(f"Right hip offset: {data.xpos[model.body('right_link_1').id] - data.xpos[body_id]}")
print("X = Red (forward), Y = Green (right), Z = Blue (up)")
print("=" * 50)


def _add_axis(user_scn, origin, direction, rgba, label):
    """Draw a capsule axis + sphere at tip."""
    d = direction / np.linalg.norm(direction)
    mid = origin + d * (AXIS_LENGTH / 2)
    tip = origin + d * AXIS_LENGTH

    # Build orthonormal basis: pick ref not parallel to d
    ref = np.array([0, 0, 1]) if abs(d[2]) > 0.9 else np.array([1, 0, 0])
    y = np.cross(d, ref)
    yn = np.linalg.norm(y)
    if yn < 1e-10:
        ref = np.array([0, 1, 0])
        y = np.cross(d, ref)
        yn = np.linalg.norm(y)
    y /= max(yn, 1e-10)
    z = np.cross(d, y)
    z /= max(np.linalg.norm(z), 1e-10)
    rot = np.column_stack([d, y, z]).ravel()

    # Axis capsule
    mujoco.mjv_initGeom(
        user_scn.geoms[user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=np.array([AXIS_RADIUS, AXIS_LENGTH / 2, 0]),
        pos=mid,
        mat=rot,
        rgba=np.array(rgba),
    )
    user_scn.ngeom += 1

    # Tip sphere
    mujoco.mjv_initGeom(
        user_scn.geoms[user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([AXIS_RADIUS * 2, 0, 0]),
        pos=tip,
        mat=np.eye(3).ravel(),
        rgba=np.array(rgba),
    )
    user_scn.ngeom += 1


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pos = data.xpos[body_id].copy()
        xmat = data.xmat[body_id].reshape(3, 3).copy()

        body_x = xmat[:, 0]  # forward
        body_y = xmat[:, 1]  # right
        body_z = xmat[:, 2]  # up

        viewer.user_scn.ngeom = 0

        # Origin sphere (white)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.025, 0, 0]),
            pos=pos,
            mat=np.eye(3).ravel(),
            rgba=np.array([1, 1, 1, 0.95]),
        )
        viewer.user_scn.ngeom += 1

        # X axis — Red (forward)
        _add_axis(viewer.user_scn, pos, body_x, [1.0, 0.10, 0.10, 0.85], "X")
        # Y axis — Green (right)
        _add_axis(viewer.user_scn, pos, body_y, [0.10, 1.0, 0.10, 0.85], "Y")
        # Z axis — Blue (up)
        _add_axis(viewer.user_scn, pos, body_z, [0.10, 0.30, 1.0, 0.85], "Z")

        viewer.sync()
        time.sleep(0.01)
