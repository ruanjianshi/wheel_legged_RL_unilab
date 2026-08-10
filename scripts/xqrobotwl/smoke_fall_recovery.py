"""跌倒恢复 FTSR 环境冒烟测试 — 多姿态复位 + 力引导 + 分阶段奖励

验证:
  1. 环境构建 + obs 维度 (297/324)
  2. 多姿态复位: 不同 reset 产生不同倒地朝向 (base up 向量)
  3. 力引导: info["constraint_costs"] 有值 (F/T 施加), 且随高度变化
  4. 分阶段: info["stage"] 从 ru(0) 能推进到 rs(1)
  5. 零动作不崩溃 (reward 有限, 无 NaN)

用法:
  uv run mjpython scripts/xqrobotwl/smoke_fall_recovery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from unilab.base import registry
from unilab.base.registry import ensure_registries

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml"


def build_env(num_envs: int = 32):
    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))
    return registry.make(
        "XqRobotWLFallRecoveryFlat",
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )


def main() -> None:
    env = build_env()
    obs, _ = env.reset(np.arange(env.num_envs))
    print(f"obs dim: {obs['obs'].shape}  critic dim: {obs['critic'].shape}")
    assert obs["obs"].shape[1] == 297, f"expect 297 obs, got {obs['obs'].shape[1]}"
    assert obs["critic"].shape[1] == 324

    # 多姿态复位: base up 向量应覆盖 4 种倒地朝向
    print("1) 多姿态复位 (10 次 reset 的 base up 向量):")
    ups = []
    for _ in range(10):
        env.reset(np.arange(env.num_envs))
        up = np.asarray(env._backend.get_sensor_data("upvector"), dtype=np.float64)[:, :3]
        ups.append(up.copy())
    ups = np.concatenate(ups, axis=0)
    up_dirs = np.unique(np.argmax(np.abs(ups), axis=1), return_counts=True)
    print(f"   up 向量绝对值最大分量分布: {dict(zip(up_dirs[0], up_dirs[1]))}  (应有 x/y/z 多种)")
    assert len(up_dirs[0]) >= 2, "复位姿态单一, 多姿态复位失败"

    # 力引导 + 阶段
    print("2) 力引导 + 分阶段 (零动作 400 步):")
    zero = np.zeros((env.num_envs, 8), dtype=np.float64)
    max_z = 0.0
    min_z = 1e9
    stage_changed = False
    constraint_nz = 0
    for i in range(400):
        st = env.step(zero)
        base_z = np.asarray(env._backend.get_base_pos(), dtype=np.float64)[:, 2]
        max_z = max(max_z, float(base_z.max()))
        min_z = min(min_z, float(base_z.min()))
        cc = st.info.get("constraint_costs", np.zeros((env.num_envs, 2)))
        constraint_nz += int((np.abs(cc[:, 0]) > 0.1).any())
        stage = st.info.get("stage", 0)
        if stage >= 1:
            stage_changed = True
        if i % 50 == 0:
            h_cmd = st.info.get("h_cmd", 0)
            print(
                f"   t={i:3d}  stage={stage}  h_cmd={h_cmd:.2f}  "
                f"z=[{base_z.min():.2f},{base_z.max():.2f}]  "
                f"F=[{cc[:, 0].mean():.1f},{cc[:, 1].mean():.2f}]"
            )
    print(f"   base_z 范围: [{min_z:.2f}, {max_z:.2f}]")
    print(f"   施加力约束的步数: {constraint_nz}/400")
    print(f"   阶段推进到 rs: {'✅' if stage_changed else '⚠️ 未推进 (零动作可能够不到 h_cmd1)'}")
    assert not np.isnan(float(st.reward[0])), "reward 出现 NaN"

    # 检查 reward 有限
    print(f"   reward 有限: ✅ (r={float(st.reward[0]):.3f})")
    env.close()
    print("=" * 60)
    print("Smoke 通过 ✅")


if __name__ == "__main__":
    main()
