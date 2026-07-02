"""MuJoCo playback of XqRobotV2 toe-walk reference trajectory.

Generates sinusoidal reference joint trajectories using pinocchio FK validation,
then plays them back in MuJoCo viewer for visual gait inspection.

Usage:
    conda activate unilab
    cd /home/robot/xiaoq/wheel_legged_RL_unilab
    python tools/pinocchio_traj/play_mujoco_toe_walk.py
"""
from __future__ import annotations

import numpy as np
import mujoco
import mujoco.viewer
import time
import os, sys

# ── Config ──
CYCLE_TIME = 0.5       # gait cycle (s)
REF_SCALE = 0.12       # trajectory amplitude
NUM_CYCLES = 6         # playback duration in cycles
PLAY_SPEED = 0.27       # 慢速看清每帧

# ── Load MuJoCo model ──
SCENE_PATH = os.path.expanduser(
    "~/xiaoq/wheel_legged_RL_unilab/src/unilab/assets/robots/xqrobotV2/scene_flat.xml"
)
model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

# Disable gravity for floating pose check
model.opt.gravity[:] = 0.0
print("Gravity: OFF (floating pose check)")

# Reset base to visible height
data.qpos[:7] = [0, 0, 0.65, 1, 0, 0, 0]
mujoco.mj_forward(model, data)

# ── Reference trajectory generator ──
mode = 1  # Option B: support hip outward, lift hip neutral
# Option A: 0=current, 1=flip, 2=both展, 3=both收
mode_names = ["A: L收R展", "B: L展R收", "C: both展", "D: both收"]

def generate_toe_walk_ref(t, cycle_time=CYCLE_TIME, ref_scale=REF_SCALE):
    phase = t / cycle_time
    sin_pos = np.sin(2 * np.pi * phase)
    cos_pos = np.cos(2 * np.pi * phase)
    T = 0.4
    L_swing = np.clip((-sin_pos - T) / (1.0 - T), 0, 1)
    R_swing = np.clip((sin_pos - T) / (1.0 - T), 0, 1)
    L_lift  = np.clip((-cos_pos - T) / (1.0 - T), 0, 1)
    R_lift  = np.clip((cos_pos - T) / (1.0 - T), 0, 1)
    lean_L  = np.clip((cos_pos - 0.2) / 0.8, 0, 1)  # right leg lifts → need weight on LEFT
    lean_R  = np.clip((-cos_pos - 0.2) / 0.8, 0, 1)
    
    # L/R hip OPPOSITE directions (per calibration)
    d = ref_scale * 1.0
    L_hip = 0.1 - lean_L * d + lean_R * d
    R_hip = 0.1 + lean_L * d - lean_R * d
    
    return np.array([
        L_hip,                              # L_hip
        0.1 + L_swing * ref_scale * 0.5,   # L_thigh
        -0.1 - L_lift * ref_scale * 5,       # L_calf
        0.0,                                 # L_wheel
        R_hip,                              # R_hip
        0.1 + R_swing * ref_scale * 0.5,   # R_thigh
        -0.1 - R_lift * ref_scale * 5,       # R_calf
        0.0,                                 # R_wheel
    ])


# ── Print wheel heights at key phases ──
print("Wheel height analysis (MuJoCo FK):")
for label, t_ref in [("Default (0.0s)", 0.0), ("Left lift (0.125s)", 0.125), 
                      ("Right lift (0.375s)", 0.375)]:
    ref = generate_toe_walk_ref(t_ref)
    data.qpos[7:15] = ref
    mujoco.mj_forward(model, data)
    lw = data.body("left_link_wheel").xpos
    rw = data.body("right_link_wheel").xpos
    print(f"  {label:20s}  L_wheel_z={lw[2]:.4f}  R_wheel_z={rw[2]:.4f}")

# ── Interactive MuJoCo viewer playback ──
print("\nCtrl+C to stop, ↑↓ to change speed, mouse to rotate view")
print(f"Playing {NUM_CYCLES} cycles at {PLAY_SPEED}x speed...\n")

# Precompute reference for smooth playback
ctrl_dt = 0.01  # 100Hz control updates
total_steps = int(NUM_CYCLES * CYCLE_TIME / ctrl_dt)
ref_angles = np.zeros((total_steps, 8))
for i in range(total_steps):
    t = i * ctrl_dt
    ref_angles[i] = generate_toe_walk_ref(t)

# ── Viewer callback ──
pause = False
speed = PLAY_SPEED
gravity_on = False
step = 0

def key_callback(keycode):
    global pause, speed, gravity_on, mode
    if keycode == 32:  # space
        pause = not pause
        print(f"{'⏸ PAUSED' if pause else '▶ PLAYING'}")
    elif keycode == 265:  # up arrow
        speed = min(speed * 1.5, 10.0)
        print(f"Speed: {speed:.1f}x")
    elif keycode == 264:  # down arrow
        speed = max(speed / 1.5, 0.1)
        print(f"Speed: {speed:.1f}x")
    elif keycode == ord('G'):
        gravity_on = not gravity_on
        model.opt.gravity[:] = [0, 0, -9.81] if gravity_on else [0, 0, 0]
        print(f"Gravity: {'ON' if gravity_on else 'OFF (floating)'}")
    elif keycode == ord('M'):
        mode = (mode + 1) % 4
        print(f"Mode: {mode_names[mode]}")

# Passive viewer for smooth rendering
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    t_start = time.time()
    cycle_count = 0
    last_printed_cycle = -1
    
    while viewer.is_running():
        elapsed = time.time() - t_start
        step = int(elapsed * speed / ctrl_dt) % total_steps if not pause else step
        
        if not pause:
            # Direct qpos set (no gravity, no physics needed)
            data.qpos[7:15] = ref_angles[step]
            mujoco.mj_forward(model, data)
            step = (step + 1) % total_steps
        
        # Cycle counter
        current_cycle = int(elapsed * speed / CYCLE_TIME)
        if current_cycle != last_printed_cycle and not pause:
            ref = ref_angles[step]
            t_sim = step * ctrl_dt
            phase = t_sim / CYCLE_TIME
            sin_val = np.sin(2*np.pi*phase)
            cos_val = np.cos(2*np.pi*phase)
            ll = np.clip((cos_val - 0.2) / 0.8, 0, 1)
            lr = np.clip((-cos_val - 0.2) / 0.8, 0, 1)
            ref_check = generate_toe_walk_ref(t_sim, CYCLE_TIME, REF_SCALE)
            print(f"  C{current_cycle} t={t_sim:.3f}s cos={cos_val:+.1f} | "
                  f"lean_L={ll:.2f} lean_R={lr:.2f} | "
                  f"L={ref[0]:+.3f} R={ref[4]:+.3f} | "
                  f"chk_L={ref_check[0]:+.3f} chk_R={ref_check[4]:+.3f}")
            last_printed_cycle = current_cycle
        
        viewer.sync()
        time.sleep(ctrl_dt * 0.1)  # Small sleep to prevent busy-waiting
