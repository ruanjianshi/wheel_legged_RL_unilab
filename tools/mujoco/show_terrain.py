"""Show xqrobotwl rough terrain in MuJoCo viewer.

Uses exact same terrain config as XqRobotWLWalkRough training.

Usage:
    uv run tools/mujoco/show_terrain.py
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SRC = os.path.expanduser("~/xiaoq/wheel_legged_RL_unilab/src")
sys.path.insert(0, SRC)

import mujoco
import mujoco.viewer

from unilab.envs.locomotion.xqrobotwl.rough import XqRobotWLRoughTerrainCfg
from unilab.terrains import TerrainGenerator

cfg = XqRobotWLRoughTerrainCfg()
gen = TerrainGenerator(cfg)
result = gen.generate()
hf = result.heights_yx

print(f"Terrain: {hf.shape[1]}x{hf.shape[0]}, z=[{hf.min():.3f}, {hf.max():.3f}]")
ts = result.terrain_origins
print(f"Cells: {ts.shape[0]}x{ts.shape[1]}")
nz = np.count_nonzero(hf)
print(f"Non-flat: {nz}/{hf.size}")
for name, st in cfg.sub_terrains.items():
    if st.proportion > 0:
        detail = ""
        if hasattr(st, "step_height_range"):
            detail = f" step_h={st.step_height_range} w={st.step_width}m"
        elif hasattr(st, "noise_range"):
            detail = f" noise={st.noise_range}"
        elif hasattr(st, "slope_range"):
            detail = f" slope={st.slope_range}"
        elif hasattr(st, "amplitude_range"):
            detail = f" amp={st.amplitude_range}"
        print(f"  {name} ({st.proportion * 100:.0f}%):{detail}")

tmpdir = tempfile.mkdtemp(prefix="xq_terrain_")
png_path = os.path.join(tmpdir, "hfield.png")
result.write_png(Path(png_path))

robot_dir = os.path.expanduser("~/xiaoq/wheel_legged_RL_unilab/src/unilab/assets/robots/xqrobotwl")
hsize = result.hfield_size
gpos = result.geom_pos

scene_xml = f"""<mujoco model="xqrobotwl rough terrain">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <asset>
    <hfield name="terrain_hfield" file="{png_path}"
      size="{hsize[0]} {hsize[1]} {hsize[2]} {hsize[3]}"/>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.25 0.30 0.35" rgb2="0.15 0.20 0.25"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
      texrepeat="10 10" reflectance="0.15"/>
  </asset>

  <worldbody>
    <light pos="0 0 15" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light pos="10 -10 8" dir="-0.5 0.5 -0.4" directional="false"
      diffuse="0.4 0.4 0.4" specular="0.3 0.3 0.3"/>
    <light pos="-10 10 6" dir="0.5 -0.5 -0.3" directional="false"
      diffuse="0.5 0.5 0.5" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="hfield" hfield="terrain_hfield"
      pos="{gpos[0]} {gpos[1]} {gpos[2]}" material="groundplane"/>
  </worldbody>

  <include file="xqrobotwl.xml"/>

  <keyframe>
    <key name="home" qpos="
        0 0 0.65  1 0 0 0
        0.1 0.15 0.15 0
        -0.1 -0.15 -0.15 0"
      ctrl="0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>"""

scene_path = os.path.join(robot_dir, "_terrain_preview.xml")
os.makedirs(robot_dir, exist_ok=True)
with open(scene_path, "w") as f:
    f.write(scene_xml)

model = mujoco.MjModel.from_xml_path(scene_path)
os.remove(scene_path)

data = mujoco.MjData(model)
model.opt.gravity[:] = [0, 0, -9.81]

key_id = model.key("home").id
mujoco.mj_resetDataKeyframe(model, data, key_id)
data.qpos[0] = float(ts[0, 0, 0])
data.qpos[1] = float(ts[0, 0, 1])
mujoco.mj_forward(model, data)

print(
    f"\nRobot at: x={data.qpos[0]:.1f} y={data.qpos[1]:.1f} z={data.body('base_link').xpos[2]:.2f}"
)
print("W/A/S/D: move camera | Space: pause | Close window to exit.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)

import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
print("Done.")
