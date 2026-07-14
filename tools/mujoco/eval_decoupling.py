"""Vx/Vy decoupling test for XqRobotV2 flat walk model.

Usage:
    uv run tools/mujoco/eval_decoupling.py [checkpoint_iter]
"""

import os
import sys

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rsl_rl.modules.mlp import MLP

from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.xqrobotV2.joystick import (
    XqRobotRewardConfig,
    XqRobotV2WalkFlatCfg,
    XqRobotV2WalkFlatEnv,
)

RUN = "2026-07-01_13-55-35_mujoco"
CHECKPOINT = int(sys.argv[1]) if len(sys.argv) > 1 else None

cfg = XqRobotV2WalkFlatCfg()
cfg.control_config.action_scale = 0.5
cfg.control_config.wheel_action_scale = 10.0
cfg.control_config.clip_actions = 100.0
cfg.commands = Commands(
    vel_limit=[[-0.6, -0.3, -1.0, -0.1, 0.45], [0.6, 0.3, 1.0, 0.1, 0.85]],
    resampling_time=999.0,
)
cfg.reward_config = XqRobotRewardConfig(
    scales={
        "tracking_lin_vel": 1.5,
        "tracking_ang_vel": 1.5,
        "lin_vel_z": -0.2,
        "ang_vel_xy": -0.02,
        "base_height": -5.0,
        "orientation": -10.0,
        "joint_action_rate": -0.1,
        "wheel_action_rate": -0.005,
        "similar_calf": -1.0,
        "hip_roll": -2.0,
        "wheel_symmetry": -0.5,
        "tsk": -2.0,
        "feet_distance": -1.0,
        "alive": 1.0,
    },
    tracking_sigma=0.3,
    base_height_target=0.65,
)
cfg.domain_rand.randomize_init_yaw = False
cfg.domain_rand.randomize_base_mass = False
cfg.domain_rand.randomize_ground_friction = False
cfg.domain_rand.randomize_kp = False
cfg.domain_rand.randomize_kd = False
cfg.domain_rand.random_com = False
cfg.curriculum.enabled = False

env = XqRobotV2WalkFlatEnv(cfg, num_envs=1, backend_type="mujoco")
obs_dim = env.obs_groups_spec["obs"]

# Load model
log_dir = os.path.expanduser(
    f"~/xiaoq/wheel_legged_RL_unilab/logs/rsl_rl_ppo/XqRobotV2WalkFlat/{RUN}"
)
if CHECKPOINT:
    ckpt_path = os.path.join(log_dir, f"model_{CHECKPOINT}.pt")
else:
    pts = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    ckpt_path = os.path.join(log_dir, pts[-1])
    CHECKPOINT = int(pts[-1].split("_")[1].split(".")[0])

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
mlp_state = {k[4:]: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")}
mlp = MLP(obs_dim, 8, [512, 512, 256, 128], activation="elu")
mlp.load_state_dict(mlp_state)
mlp.eval()

EVAL_TIME = 5.0
WARMUP = 1.5
CRTL_DT = 0.01
SKIP = int(WARMUP / CRTL_DT)
N = int(EVAL_TIME / CRTL_DT)

tests = [
    ("Fwd  vx=+0.6", [0.6, 0.0, 0.0, 0.0, 0.65]),
    ("Fwd  vx=+0.3", [0.3, 0.0, 0.0, 0.0, 0.65]),
    ("Fwd  vx=-0.3", [-0.3, 0.0, 0.0, 0.0, 0.65]),
    ("Lat  vy=+0.3", [0.0, 0.3, 0.0, 0.0, 0.65]),
    ("Lat  vy=-0.3", [0.0, -0.3, 0.0, 0.0, 0.65]),
    ("Comb vx=0.3,vy=0.2", [0.3, 0.2, 0.0, 0.0, 0.65]),
]

print(f"\n{'=' * 78}")
print(f"Model: iter={CHECKPOINT}  Warmup={WARMUP}s  Eval={EVAL_TIME}s")
print(f"{'=' * 78}")
print(
    f"{'Test':<26} {'CmdVx':>6} {'CmdVy':>6} {'AvgVx':>7} {'AvgVy':>7} {'VxErr':>7} {'VyXtalk':>8}"
)
print("-" * 78)

for name, cmd in tests:
    cmd_arr = np.array([cmd], dtype=np.float32)
    total_steps = N + SKIP

    # Start fresh each test
    env._state = None
    env.init_state()
    state = env._state

    # Inject target command into info
    state.info["commands"] = cmd_arr

    vx_buf, vy_buf = [], []

    for s in range(total_steps):
        # Keep injecting commands (prevent resampling or overwrite)
        env._state.info["commands"] = cmd_arr

        # Policy inference
        with torch.inference_mode():
            raw_act = mlp(torch.from_numpy(env._state.obs["obs"])).numpy()

        # Use env.step which handles apply_action + physics + update_state + autoreset
        state = env.step(raw_act)

        # After autoreset (if robot died), re-inject commands
        env._state.info["commands"] = cmd_arr

        if s >= SKIP:
            lv = env.get_local_linvel()
            vx_buf.append(float(lv[0, 0]))
            vy_buf.append(float(lv[0, 1]))

    avg_vx = np.mean(vx_buf)
    avg_vy = np.mean(vy_buf)
    vx_err = abs(avg_vx - cmd[0])
    vy_xtalk = abs(avg_vy - cmd[1])

    print(
        f"{name:<26} {cmd[0]:>6.2f} {cmd[1]:>6.2f} {avg_vx:>7.3f} {avg_vy:>7.3f} {vx_err:>7.3f} {vy_xtalk:>8.3f}"
    )

print("-" * 78)
print("VyXtalk ≤ 0.05 = good decoupling | > 0.1 = coupling present\n")

env.close()
