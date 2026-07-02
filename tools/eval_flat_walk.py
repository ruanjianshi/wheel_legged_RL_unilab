"""Evaluate trained model with test velocity commands."""
import sys, numpy as np, torch, os
sys.path.insert(0, '/home/robot/xiaoq/wheel_legged_RL_unilab/src')
os.chdir('/home/robot/xiaoq/wheel_legged_RL_unilab')

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.training import ensure_registries, BackendAdapter
ensure_registries()

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir='/home/robot/xiaoq/wheel_legged_RL_unilab/conf/ppo', version_base='1.3'):
    cfg = compose(config_name='config', overrides=['task=xqrobotV2_walk_flat/mujoco'])

adapter = BackendAdapter(cfg, root_dir='/home/robot/xiaoq/wheel_legged_RL_unilab', algo_name='ppo')
from unilab.base import registry
env = registry.make('XqRobotV2WalkFlat', num_envs=1, sim_backend='mujoco',
                    env_cfg_override=adapter.build_task_env_cfg_override())

ckpt = torch.load('logs/rsl_rl_ppo/XqRobotV2WalkFlat/2026-06-30_16-09-51_mujoco/model_4999.pt', map_location='cpu')

import torch.nn as nn
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(297, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 8),
        )
    def forward(self, x):
        return self.mlp(x)

actor = Actor()
actor.load_state_dict({k:v for k,v in ckpt['actor_state_dict'].items() if 'distribution' not in k})
actor.eval()

tests = {
    'stand':    [0.0, 0.0, 0.0, 0.0, 0.65],
    'fwd_0.3':  [0.3, 0.0, 0.0, 0.0, 0.65],
    'fwd_0.6':  [0.6, 0.0, 0.0, 0.0, 0.65],
    'back':     [-0.3,0.0, 0.0, 0.0, 0.65],
    'L_turn':   [0.0, 0.0, 0.5, 0.0, 0.65],
    'R_turn':   [0.0, 0.0, -0.5,0.0, 0.65],
}

print(f"{'Command':<10} {'Vx':>7} {'Vy':>7} {'tilt°':>7} {'H(m)':>7} | {'L_th':>7} {'R_th':>7} {'L_c':>7} {'R_c':>7}")
print('-' * 85)

for name, cmd in tests.items():
    obs, info = env.reset(np.array([0]))
    for _ in range(80):
        info['commands'][:] = cmd
        env._update_commands(info)
        linvel = env.get_local_linvel()
        grav = env._backend.get_sensor_data(env._cfg.sensor.upvector)
        pos = env.get_dof_pos()
        vel = env.get_dof_vel()
        obs_dict = env._compute_obs(info, linvel, env.get_gyro(), grav, pos, vel)
        with torch.no_grad():
            a = actor(torch.tensor(obs_dict['obs'].reshape(1, -1), dtype=torch.float32)).numpy()
        env.step(a)

    tilt = np.rad2deg(np.arccos(np.clip(grav[0, 2], -1, 1)))
    dof = pos[0]
    lv = linvel[0]
    hz = env._base_height_values(1)[0]
    print(f'{name:<10} {lv[0]:+7.3f} {lv[1]:+7.3f} {tilt:7.1f} {hz:7.3f} | {dof[1]:+7.2f} {dof[4]:+7.2f} {dof[2]:+7.2f} {dof[5]:+7.2f}')

print('\nOK: stand→Vx≈0; fwd→Vx>0; back→Vx<0; turn→Vyaw≠0')
