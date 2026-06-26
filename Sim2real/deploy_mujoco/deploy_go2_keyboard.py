import time
import re

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR


paused = False
ctrl_f = [0, 0]  # [ctrl_pressed, any_key_pressed]
cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def key_callback(keycode):
    global paused
    if chr(keycode) == ' ':
        paused = not paused


def on_press(key):
    ctrl_f[1] = 1
    if key == keyboard.Key.up:
        cmd[0] = 1.0
    elif key == keyboard.Key.down:
        cmd[0] = -1.0

    if key == keyboard.Key.left:
        if ctrl_f[0] == 0:
            cmd[1] = 1.0
        else:
            cmd[2] = 1.0
    elif key == keyboard.Key.right:
        if ctrl_f[0] == 0:
            cmd[1] = -1.0
        else:
            cmd[2] = -1.0

    if key == keyboard.Key.ctrl:
        ctrl_f[0] = 1


def on_release(key):
    ctrl_f[1] = 0
    if key == keyboard.Key.up or key == keyboard.Key.down:
        cmd[0] = 0.0

    if key == keyboard.Key.left or key == keyboard.Key.right:
        cmd[1] = 0.0
        cmd[2] = 0.0

    if key == keyboard.Key.ctrl:
        ctrl_f[0] = 0
        cmd[2] = 0.0


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def infer_policy_input_dim(policy, default_dim):
    """Infer TorchScript policy input dimension from dry-run error message."""
    try:
        _ = policy(torch.zeros((1, default_dim), dtype=torch.float32))
        return default_dim
    except RuntimeError as e:
        msg = str(e)
        match = re.search(r"\((\d+)x(\d+) and (\d+)x(\d+)\)", msg)
        if match:
            return int(match.group(3))
        raise RuntimeError(
            f"Failed to infer policy input dimension. Please check model/config. Original error: {msg}"
        )


if __name__ == "__main__":
    import argparse
    from pynput import keyboard

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd[:] = np.array(config["cmd_init"], dtype=np.float32)

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)
    policy_input_dim = infer_policy_input_dim(policy, num_obs)
    if policy_input_dim % num_obs != 0:
        raise ValueError(
            f"Unsupported policy input dim {policy_input_dim}. It must be a multiple of num_obs={num_obs}."
        )
    policy_obs = np.zeros(policy_input_dim, dtype=np.float32)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if not paused:
                tau = pd_control(target_dof_pos, data.qpos[7:], kps, np.zeros_like(kds), data.qvel[6:], kds)
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)

                counter += 1
                if counter % control_decimation == 0:
                    qj = data.qpos[7:]
                    dqj = data.qvel[6:]
                    quat = data.qpos[3:7]
                    lin_vel = data.qvel[:3]
                    ang_vel = data.qvel[3:6]

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    lin_vel = lin_vel * lin_vel_scale
                    ang_vel = ang_vel * ang_vel_scale

                    obs[:3] = lin_vel
                    obs[3:6] = ang_vel
                    obs[6:9] = gravity_orientation
                    obs[9:12] = cmd * cmd_scale
                    obs[12:12 + num_actions] = qj
                    obs[12 + num_actions:12 + 2 * num_actions] = dqj
                    obs[12 + 2 * num_actions:12 + 3 * num_actions] = action

                    if policy_input_dim == num_obs:
                        policy_in = obs
                    else:
                        policy_obs[:-num_obs] = policy_obs[num_obs:]
                        policy_obs[-num_obs:] = obs
                        policy_in = policy_obs

                    obs_tensor = torch.from_numpy(policy_in).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    target_dof_pos = action * action_scale + default_angles

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
