"""Show XqRobotV2 stairs-only terrain (NP3O config) in MuJoCo viewer.

Usage:
    uv run tools/mujoco/show_stairs.py
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

from unilab.envs.locomotion.xqrobotV2.stairs import StairsOnlyTerrainCfg
from unilab.terrains import TerrainGenerator

cfg = StairsOnlyTerrainCfg()
gen = TerrainGenerator(cfg)
result = gen.generate()
hf = result.heights_yx

print(f"Terrain: {hf.shape[1]}x{hf.shape[0]}, z=[{hf.min():.3f}, {hf.max():.3f}]")
print(f"Cells: {result.terrain_origins.shape[0]}x{result.terrain_origins.shape[1]}")
print(f"Sub-terrains: {list(cfg.sub_terrains.keys())}")
for name, st in cfg.sub_terrains.items():
    print(f"  {name}: step_height={st.step_height_range}, step_width={st.step_width}m, prop={st.proportion}")
hsize = result.hfield_size

tmpdir = tempfile.mkdtemp(prefix="xq_stairs_")
png_path = os.path.join(tmpdir, "hfield.png")
result.write_png(Path(png_path))

robot_dir = os.path.expanduser("~/xiaoq/wheel_legged_RL_unilab/src/unilab/assets/robots/xqrobotV2")
gpos = result.geom_pos
ts = result.terrain_origins

scene_xml = f"""<mujoco model="xqrobotV2 stairs terrain">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>

  <asset>
    <hfield name="terrain_hfield" file="{png_path}"
      size="{hsize[0]} {hsize[1]} {hsize[2]} {hsize[3]}"/>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.30 0.35 0.25" rgb2="0.20 0.25 0.15"
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

  <include file="xqrobotV2.xml"/>

  <keyframe>
    <key name="home" qpos="
        0 0 0.80  1 0 0 0
        -0.1 0.1 -0.1 0
        0.1 0.1 -0.1 0"
      ctrl="0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>"""

scene_path = os.path.join(robot_dir, "_stairs_preview.xml")
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

print(f"Robot at: x={data.qpos[0]:.1f} y={data.qpos[1]:.1f} z={data.body('base_link').xpos[2]:.2f}")
print("W/A/S/D: move camera | Space: pause | Close window to exit.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)

import shutil

shutil.rmtree(tmpdir, ignore_errors=True)
print("Done.")
