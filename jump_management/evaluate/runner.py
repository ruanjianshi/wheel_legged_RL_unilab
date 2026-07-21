"""跳跃评估 runner — 加载模型, 运行指定场景, 记录结果."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from jump_management.evaluate.scenarios import (
    JumpCycleRecord,
    EvalResult,
    generate_jump_commands,
)

# 将项目根目录加入 path
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _get_scenario_vx(scenario: str) -> float:
    from jump_management.evaluate.scenarios import FIXED_DIST_SCENARIOS, PLATFORM_SCENARIO
    if scenario in FIXED_DIST_SCENARIOS:
        return FIXED_DIST_SCENARIOS[scenario]["vx"]
    if scenario == "random":
        return 0.5  # 先取中值
    if scenario == "platform":
        return PLATFORM_SCENARIO["vx"]
    return 0.3


def eval_one_scenario(
    model_path: str,
    task_name: str,
    scenario: str,
    num_episodes: int = 50,
    out_path: str | None = None,
) -> list[dict]:
    """评估单个模型在单个场景上的表现.

    Args:
        model_path: .pt checkpoint 路径
        task_name: "ppo_only" | "srl_full" | ...
        scenario: "fix_01m" | "fix_02m" | "fix_03m" | "random" | "platform"
        num_episodes: 每场景评估回合数
        out_path: JSON 输出路径 (可选)

    Returns:
        list of EvalResult dicts
    """
    # 映射 task_name -> UniLab task
    if task_name == "ppo_only":
        unilab_task = "XqRobotWLJumpFlat"
    else:
        unilab_task = "XqRobotWLJumpSRLFlat"

    # 加载模型
    import torch
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # 创建环境 (无DR, 单env)
    from unilab.base.registry import make
    env = make(unilab_task, "mujoco", num_envs=1)
    env.reset()

    # 加载模型参数
    actor_state = checkpoint.get("actor_state_dict", checkpoint.get("model_state_dict"))
    if "actor_state_dict" in checkpoint:
        # 仅加载 actor
        from unilab.algos.torch.ppo.network import ActorCritic
        # 根据观测维度构建网络
        obs_dim = env.obs_groups_spec["obs"]
        action_dim = 8
        model = ActorCritic(
            obs_dim=obs_dim,
            critic_obs_dim=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=[512, 512, 256, 128],
            critic_hidden_dims=[512, 512, 256, 128],
            activation="elu",
            init_noise_std=0.3,
        )
        model.actor.load_state_dict(actor_state)
        model.eval()
    else:
        model = None
        actor_state_dict = checkpoint["model_state_dict"]

    results = []

    for ep in range(num_episodes):
        obs_dict, info = env.reset()
        survived = True
        jump_cycles: list[JumpCycleRecord] = []
        current_jump: JumpCycleRecord | None = None
        last_fsm = -1
        ep_height_history: list[float] = []
        ep_x_history: list[float] = []

        vx_target = _get_scenario_vx(scenario)
        if scenario == "random":
            vx_target = float(np.random.uniform(0.0, 1.0))

        for step in range(1000):
            # 覆盖命令
            cmds = np.array([[vx_target, 0.0, 0.0, 0.0,
                             1.0 if (step % 200) < 100 else 0.0]], dtype=np.float64)
            info["commands"] = cmds

            # 推理
            if "obs" in obs_dict:
                if model is not None:
                    with torch.no_grad():
                        obs_t = torch.from_numpy(obs_dict["obs"]).float()
                        action_t = model.act(obs_t, deterministic=True)
                        action = action_t.detach().cpu().numpy()
                else:
                    action = np.zeros((1, 8), dtype=np.float64)
            else:
                action = np.zeros((1, 8), dtype=np.float64)

            obs_dict, reward, terminated, truncated, info = env.step(action)
            if isinstance(obs_dict, tuple):
                obs_dict = obs_dict[0]
                if len(obs_dict) > 1:
                    info = obs_dict[1]

            # 记录轨迹
            if hasattr(env, 'get_dof_pos'):
                pass  # 简化处理
            ep_height_history.append(float(info.get("base_height", 0)))
            ep_x_history.append(float(info.get("base_x", 0)))

            # FSM 检测跳跃周期
            fsm = int(info.get("fsm_state", [-1])[0] if isinstance(info.get("fsm_state"), np.ndarray) else -1)
            if fsm == 2 and current_jump is None:
                current_jump = JumpCycleRecord(
                    takeoff_x=ep_x_history[-1] if ep_x_history else 0.0
                )
            if fsm >= 3 and current_jump is not None:
                current_jump.landing_x = ep_x_history[-1] if ep_x_history else current_jump.takeoff_x
                current_jump.max_height = max(ep_height_history[-50:]) if ep_height_history else 0.0
                current_jump.duration_steps = 100  # rough estimate
                # 轮滑: wheel_vel * r - base_vx
                current_jump.wheel_slip_at_landing = float(
                    info.get("wheel_slip", [0])[0] if isinstance(info.get("wheel_slip"), np.ndarray) else 0
                )
                jump_cycles.append(current_jump)
                current_jump = None
            last_fsm = fsm

            if bool(terminated) or bool(truncated):
                survived = not bool(terminated)
                break
        else:
            survived = True

        result = EvalResult(scenario=scenario, survived=survived, jump_cycles=jump_cycles)
        results.append(result.to_dict())

    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results
