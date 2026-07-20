"""评估 Wheeled-SRL 跳跃策略 — 通过Hydra加载配置运行"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from unilab.base import registry
import unilab.envs.locomotion.xqrobotwl.jump  # register env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_base', type=str,
        default='logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-07-14_17-19-55_mujoco')
    parser.add_argument('--num_episodes', type=int, default=30)
    parser.add_argument('--task', type=str, default='xqrobotwl_jump_flat')
    args = parser.parse_args()
    log_dir = args.log_base

    # 通过Hydra加载配置
    GlobalHydra.instance().clear()
    import os as _os
    from omegaconf import OmegaConf
    from unilab.training.reward import extract_reward_config
    root = _os.path.abspath('conf/ppo')
    with initialize_config_dir(version_base="1.3", config_dir=root):
        cfg = compose(
            config_name="config",
            overrides=[f"task={args.task}/mujoco"],
        )

    # 构建 env_cfg_override (含 reward_config)
    env_override = extract_reward_config(cfg)
    env_override.update(OmegaConf.to_container(cfg.env, resolve=True))

    # 创建环境
    env = registry.make(cfg.training.task_name, num_envs=1,
                        sim_backend='mujoco', env_cfg_override=env_override)
    # ★ 评估时绕过 warmup，直接允许跳跃
    env._warmup_cutoff = 0

    # 加载ONNX策略
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(log_dir, 'policy.onnx'))
    iname = sess.get_inputs()[0].name
    obs_dict, _ = env.reset(np.array([0]))

    # 运行
    data = {k: [] for k in ['t','z','vz','vx','pitch','wr','fsm','rew','cmd','wlin']}
    step = ep = 0
    while ep < args.num_episodes:
        obs = obs_dict['obs'][0]
        if isinstance(obs, (list, tuple)):
            obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        elif hasattr(obs, 'numpy'):
            obs = obs.numpy().reshape(1, -1)
        else:
            obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        a = np.clip(sess.run(None, {iname: obs})[0][0], -1, 1)
        env.step(a)
        obs_dict = env._state.obs
        bp = env._backend.get_base_pos()[0]
        lv = env.get_local_linvel()[0]
        gy = env.get_gyro()[0]
        dv = env.get_dof_vel()[0]
        data['t'].append(step * 0.01)
        data['z'].append(float(bp[2]))
        data['vz'].append(float(lv[2]))
        data['vx'].append(float(lv[0]))
        data['pitch'].append(float(np.arcsin(np.clip(-gy[1],-1,1))))
        data['wr'].append(float(dv[6:8].mean()))
        data['wlin'].append(float(dv[6:8].mean()*0.065))
        data['fsm'].append(int(env._fsm_state[0]))
        data['rew'].append(float(env._state.reward[0]))
        data['cmd'].append(float(env._state.info['commands'][0,4]))
        step += 1
        if env._state.terminated[0] or env._state.truncated[0]:
            ep += 1
            obs_dict, _ = env.reset(np.array([0]))
            print(f"Ep {ep}/{args.num_episodes}")

    # 分析跳跃
    z = np.array(data['z']); cmd = np.array(data['cmd'])
    vz = np.array(data['vz']); vx = np.array(data['vx'])
    wl = np.array(data['wlin'])
    jumps = []
    in_j = False
    for i in range(1, len(z)):
        if cmd[i]>0.5 and cmd[i-1]<=0.5 and not in_j:
            in_j=True; js=i
        elif in_j and cmd[i]<=0.5 and (i-js)>30:
            in_j=False; jumps.append((js,i))
    jumps = [(s,e) for s,e in jumps if (e-s)>40]

    heights,dists,tvz,lvz,prng,wme=[],[],[],[],[],[]
    for s,e in jumps:
        zs=z[s:e+50]; xs=np.array(data['t'])[s:e+50]
        h=zs.max()-zs[:min(20,len(zs))].mean()
        d=abs(xs[-1]-xs[0])*np.nan  # time not x - use step count * 0.01 * vel
        heights.append(h)
        dists.append(abs(np.trapezoid(np.array(data['vx'])[s:e+50], data['t'][s:e+50])))
        vz_seg=vz[s:e+50]
        tvz.append(float(vz_seg[vz_seg>0.3][0]) if len(vz_seg[vz_seg>0.3])>0 else 0)
        lvz.append(float(abs(vz_seg[-min(10,len(vz_seg)//2):].min())))
        ps=np.array(data['pitch'])[s:e+50]
        prng.append(float(np.rad2deg(ps.max()-ps.min())))
        li=min(len(wl[s:e+50])-1,(e-s)//2*2)
        wme.append(float(abs(wl[s:e+50][li]-vx[s:e+50][li])))

    print(f"\n=== {len(jumps)} jumps ===")
    print(f"Height:  {np.mean(heights):.3f}±{np.std(heights):.3f}m (max {np.max(heights):.3f})")
    print(f"Dist:    {np.mean(dists):.3f}±{np.std(dists):.3f}m")
    print(f"Takeoff: {np.mean(tvz):.2f}m/s | Landing Vz: {np.mean(lvz):.2f}m/s")
    print(f"Pitch:   {np.mean(prng):.1f}° | Wheel Match: {np.mean(wme):.3f}m/s")

    # 绘图
    t = np.array(data['t'])
    fig,axs = plt.subplots(5,1,figsize=(14,14),sharex=True)
    for s,e in jumps:
        for ax in axs[:2]:
            ax.axvspan(t[s],t[min(e,len(t)-1)],alpha=0.08,color='green')
    axs[0].plot(t,z,'b-',lw=0.8); axs[0].axhline(0.65,color='gray',ls='--',alpha=0.5)
    axs[0].set_ylabel('Height (m)'); axs[0].grid(True,alpha=0.3)
    axs[1].plot(t,vz,'g-',lw=0.8); axs[1].axhline(0,color='gray',ls=':',alpha=0.3)
    axs[1].set_ylabel('Vz (m/s)'); axs[1].grid(True,alpha=0.3)
    axs[2].plot(t,np.array(data['fsm']),'r-',lw=1.0)
    axs[2].set_ylabel('FSM'); axs[2].set_ylim(-2,6); axs[2].grid(True,alpha=0.3)
    axs[3].plot(t,np.array(data['wr']),'m-',lw=0.8)
    axs[3].set_ylabel('Wheel (rad/s)'); axs[3].grid(True,alpha=0.3)
    axs[4].plot(t,np.rad2deg(data['pitch']),'c-',lw=0.8)
    axs[4].set_ylabel('Pitch (°)'); axs[4].set_xlabel('Time (s)'); axs[4].grid(True,alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(log_dir,'eval_trajectories.png'),dpi=150)

    if jumps:
        s,e=jumps[len(jumps)//2]; sl=slice(max(0,s-30),e+80); ts=t[sl]-t[s]
        fig2,axs2=plt.subplots(4,1,figsize=(10,12),sharex=True)
        axs2[0].plot(ts,z[sl],'b-',lw=1.5); axs2[0].axhline(0.65,color='gray',ls='--',alpha=0.5)
        axs2[0].set_ylabel('Height (m)'); axs2[0].grid(True,alpha=0.3)
        axs2[1].plot(ts,vz[sl],'g-',lw=1.5); axs2[1].axhline(0,color='gray',ls=':',alpha=0.3)
        axs2[1].set_ylabel('Vz (m/s)'); axs2[1].grid(True,alpha=0.3)
        axs2[2].plot(ts,vx[sl],'c-',lw=1.5,label='Body Vx')
        axs2[2].plot(ts,wl[sl],'r--',lw=1.5,label='Wheel (rω)')
        axs2[2].set_ylabel('Fwd Vel (m/s)'); axs2[2].legend(); axs2[2].grid(True,alpha=0.3)
        axs2[3].plot(ts,np.rad2deg(data['pitch'])[sl],'c-',lw=1.5)
        axs2[3].set_ylabel('Pitch (°)'); axs2[3].set_xlabel('Time (s)'); axs2[3].grid(True,alpha=0.3)
        plt.tight_layout()
        fig2.savefig(os.path.join(log_dir,'eval_single_jump.png'),dpi=150)

    stats=dict(jump_count=len(jumps), mean_height_m=round(float(np.mean(heights)),3),
        std_height_m=round(float(np.std(heights)),3), max_height_m=round(float(np.max(heights)),3),
        mean_dist_m=round(float(np.mean(dists)),3),
        mean_takeoff_vz=round(float(np.mean(tvz)),2), mean_landing_vz=round(float(np.mean(lvz)),2),
        mean_pitch_range_deg=round(float(np.mean(prng)),1),
        mean_wheel_match_err_ms=round(float(np.mean(wme)),3))
    with open(os.path.join(log_dir,'eval_stats.json'),'w') as f:
        json.dump(stats,f,indent=2,ensure_ascii=False)
    print(f"\nStats: {json.dumps(stats,indent=2,ensure_ascii=False)}")

if __name__=='__main__':
    main()
