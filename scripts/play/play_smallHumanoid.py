#!/usr/bin/env python3
"""Interactive keyboard-controlled policy playback using GLFW rendering.

W/S: fwd/back  |  A/D: strafe  |  Q/E: turn  |  R: stop  |  ESC: quit
"""
import sys
import time
import glfw
import numpy as np
import mujoco
import torch

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from unilab.training.common import ensure_registries
from unilab.envs.locomotion.smallHumanoidRobot.joystick import (
    SmallHumanoidWalkFlatCfg, SmallHumanoidRewardConfig, SmallHumanoidWalkFlatEnv,
)


def build_actor(in_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 512), torch.nn.ELU(),
        torch.nn.Linear(512, 256), torch.nn.ELU(),
        torch.nn.Linear(256, 128), torch.nn.ELU(),
        torch.nn.Linear(128, out_dim),
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logs/rsl_rl_ppo/SmallHumanoidWalkFlat/2026-06-17_15-29-01_motrix/model_999.pt")
    args = parser.parse_args()

    print("Loading registries...", flush=True)
    ensure_registries()

    # Load model
    print("Loading checkpoint...", flush=True)
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    sd = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
    out_dim = sd["mlp.6.weight"].shape[0]
    in_dim = sd["mlp.0.weight"].shape[1]
    print(f"Model: obs={in_dim}, action={out_dim}", flush=True)

    actor = build_actor(in_dim, out_dim)
    actor.load_state_dict({k: v for k, v in sd.items() if not k.startswith("obs_normalizer")}, strict=False)
    actor.eval()

    action_std = torch.exp(sd["distribution.std_param"]).cpu().numpy() if "distribution.std_param" in sd else np.ones(out_dim) * 0.5
    obs_mean = sd["obs_normalizer._mean"].cpu().numpy() if "obs_normalizer._mean" in sd else None
    obs_std = sd["obs_normalizer._std"].cpu().numpy() if "obs_normalizer._std" in sd else None

    # Create env
    print("Creating env...", flush=True)
    cfg = SmallHumanoidWalkFlatCfg()
    cfg.reward_config = SmallHumanoidRewardConfig(
        scales={
            "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.05,
            "lin_vel_z": -0.2, "ang_vel_xy": -0.05,
            "base_height": -2.0, "orientation": -6.0,
            "action_rate": -0.005, "weighted_pose": -0.01, "alive": 0.0,
        },
        base_height_target=0.52, max_tilt_deg=20.0, min_base_height=0.22,
        pose_weights=[0.01, 1.0, 0.01, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
    )
    env = SmallHumanoidWalkFlatEnv(cfg, num_envs=1, backend_type="mujoco")
    backend = env._backend
    mj_model = backend.model

    # GLFW setup
    print("Initializing GLFW...", flush=True)
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    window = glfw.create_window(1200, 900, "Robot — W/S fwd/back A/D strafe Q/E turn R stop ESC quit", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    key_state = set()

    def key_cb(win, key, scancode, action, mods):
        if action == glfw.PRESS:
            key_state.add(key)
        elif action == glfw.RELEASE:
            key_state.discard(key)

    glfw.set_key_callback(window, key_cb)

    # MuJoCo rendering setup
    scene = mujoco.MjvScene(mj_model, maxgeom=10000)
    context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
    cam = mujoco.MjvCamera()
    cam.distance = 4.0
    cam.elevation = -15.0
    cam.azimuth = 90.0
    cam.lookat[:] = [0, 0, 0.5]
    opt = mujoco.MjvOption()

    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = backend._qpos_view[0]
    mujoco.mj_forward(mj_model, mj_data)

    # Reset env
    obs, info = env.reset(env_indices=np.array([0]))
    info["commands"] = np.array([[0.3, 0.0, 0.0]], dtype=np.float32)

    print("=" * 60, flush=True)
    print("  W/S: fwd/back  |  A/D: strafe  |  Q/E: turn  |  R: stop", flush=True)
    print("  ESC: close window to quit", flush=True)
    print("=" * 60, flush=True)

    frame = 0
    while not glfw.window_should_close(window):
        # Keyboard → velocity command
        vx = 0.0
        vy = 0.0
        va = 0.0
        if glfw.KEY_W in key_state: vx = 0.4
        if glfw.KEY_S in key_state: vx = -0.2
        if glfw.KEY_A in key_state: vy = -0.2
        if glfw.KEY_D in key_state: vy = 0.2
        if glfw.KEY_Q in key_state: va = -0.8
        if glfw.KEY_E in key_state: va = 0.8
        if glfw.KEY_R in key_state: vx = vy = va = 0.0

        info["commands"] = np.array([[vx, vy, va]], dtype=np.float32)

        # Policy inference
        obs_arr = obs["obs"].astype(np.float32)
        if obs_mean is not None:
            obs_arr = (obs_arr - obs_mean) / (obs_std + 1e-8)

        with torch.no_grad():
            mu = actor(torch.from_numpy(obs_arr)).cpu().numpy()
            action = mu + np.random.randn(*mu.shape).astype(np.float32) * action_std * 0.3
            action = np.clip(action, -1.0, 1.0)

        state = env.step(action)
        obs = state.obs
        info = state.info

        if state.terminated or state.truncated:
            obs, info = env.reset(env_indices=np.array([0]))
            info["commands"] = np.array([[vx, vy, va]], dtype=np.float32)
            print("Reset!", flush=True)
            print("Reset!", flush=True)

        # Render
        mj_data.qpos[:] = backend._qpos_view[0]
        mujoco.mj_forward(mj_model, mj_data)

        viewport_w, viewport_h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_w, viewport_h)
        mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()
        time.sleep(0.01)

        frame += 1
        if frame % 100 == 0:
            print(f"Frame {frame}  cmd=[{vx:.1f},{vy:.1f},{va:.1f}]", flush=True)

    glfw.terminate()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
