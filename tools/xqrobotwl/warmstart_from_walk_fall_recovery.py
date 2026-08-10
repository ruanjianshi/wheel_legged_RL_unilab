"""用 walk_flat 模型热启动跌倒恢复 CPO 训练 — obs 相同, 填权重保留 CPO 结构

FTSR fall_recovery 的观测 = 基础 297/324 (commands[4]=目标高度), 与 walk_flat
一致。把 walk 的 actor/critic 权重填入**最新 CPO checkpoint 模板** (保留
constraint_critic + optimizer 参数组结构, 避免 param-group 不匹配)。

用法:
  uv run tools/xqrobotwl/warmstart_from_walk_fall_recovery.py
输出: logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat/warmstart_from_walk/model_0.pt
然后 resume: algo.load_run=warmstart_from_walk
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WALK_RUNS = sorted(glob.glob(str(ROOT / "logs/rsl_rl_ppo/XqRobotWLWalkFlat/*/model_*.pt")))
if not WALK_RUNS:
    raise SystemExit("找不到 walk_flat checkpoint, 请先训练 walk")
WALK_CKPT = WALK_RUNS[-1]  # 最新 walk checkpoint

# 模板: 最新 fall_recovery CPO checkpoint (含 constraint_critic + optimizer 结构),
# 排除 warmstart_from_walk 自身
fr_runs = sorted(
    p
    for p in glob.glob(str(ROOT / "logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat/*/model_*.pt"))
    if "warmstart_from_walk" not in p
)
if not fr_runs:
    raise SystemExit("找不到 fall_recovery CPO checkpoint 模板, 请先跑一次训练")
TEMPLATE_CKPT = fr_runs[-1]
OUT = ROOT / "logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat/warmstart_from_walk/model_0.pt"


def main() -> None:
    walk = torch.load(WALK_CKPT, map_location="cpu", weights_only=False)
    tpl = torch.load(TEMPLATE_CKPT, map_location="cpu", weights_only=False)

    w_actor = walk["actor_state_dict"]
    w_critic = walk["critic_state_dict"]
    t_actor = tpl["actor_state_dict"]
    t_critic = tpl["critic_state_dict"]

    # 验证架构一致 (obs 297, 动作 8)
    assert t_actor["mlp.0.weight"].shape == w_actor["mlp.0.weight"].shape, "actor 结构不一致"
    assert t_critic["mlp.0.weight"].shape == w_critic["mlp.0.weight"].shape, "critic 结构不一致"

    # 填充 walk 权重
    t_actor.update({k: v.clone() for k, v in w_actor.items()})
    t_critic.update({k: v.clone() for k, v in w_critic.items()})

    # 重置迭代 (保留模板的 optimizer/constraint_critic 结构)
    tpl["iteration"] = 0
    tpl["iter"] = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tpl, OUT)
    print(f"✅ 热启动 checkpoint: {OUT}")
    print(f"   来源 walk: {WALK_CKPT}")
    print(f"   模板 CPO: {TEMPLATE_CKPT} (保留 constraint_critic/optimizer 结构)")
    print("   继承 walk 全部 actor/critic 权重 (两轮平衡能力)")
    print("   resume: algo.load_run=warmstart_from_walk")


if __name__ == "__main__":
    main()
