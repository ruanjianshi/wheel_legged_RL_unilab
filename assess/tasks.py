"""Task + Algorithm registry for multi-method evaluation.

Hierarchy:
    task (e.g., flat_walk) → algorithm (e.g., PPO, SAC, APPO)
    
Each (task, algo) pair maps to:
    - Log root directory
    - Default evaluation suite
    - Algorithm metadata for reporting

Usage:
    # Register a task-algorithm combination
    register("flat_walk", "ppo", ...)
    
    # Query
    get_task("flat_walk")
    get_algo("ppo")
    get_pair("flat_walk", "ppo")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).parent.parent

# Manually derived: task → known log subdirectory patterns
# Extended when new algorithms are trained
_LOG_PATTERNS = {
    ("flat_walk", "ppo"): "rsl_rl_ppo/XqRobotV2WalkFlat",
    ("toe_walk", "ppo"): "rsl_rl_ppo/XqRobotV2ToeWalkFlat",
    ("rough_walk", "ppo"): "rsl_rl_ppo/XqRobotV2WalkRough",
}


# ── Task definitions ───────────────────────────────────────────────────────

@dataclass
class TaskDef:
    """A locomotion task (problem)."""
    key: str               # e.g., "flat_walk"
    name: str              # e.g., "Flat Ground Walking"
    robot: str             # e.g., "XqRobotV2"
    default_suite: str = "full"
    cmd_dim: int = 5       # command vector dimension


@dataclass
class AlgoDef:
    """A reinforcement learning algorithm."""
    key: str               # e.g., "ppo"
    name: str              # e.g., "PPO"
    description: str = ""
    # Typical hyperparameters for reference
    defaults: dict = field(default_factory=dict)


@dataclass
class TaskAlgoPair:
    """A trained (task, algorithm) combination ready for evaluation."""
    task: TaskDef
    algo: AlgoDef
    log_root: Path
    default_suite: str

    @property
    def key(self) -> str:
        return f"{self.task.key}/{self.algo.key}"

    @property
    def display(self) -> str:
        return f"{self.task.name} ({self.algo.name})"


# ── Registries ─────────────────────────────────────────────────────────────

_tasks: dict[str, TaskDef] = {}
_algos: dict[str, AlgoDef] = {}
_pairs: dict[tuple[str, str], TaskAlgoPair] = {}


def register_task(task: TaskDef):
    _tasks[task.key] = task


def register_algo(algo: AlgoDef):
    _algos[algo.key] = algo


def register(task_key: str, algo_key: str, log_subdir: str | None = None, default_suite: str | None = None):
    """Register a trained task+algo combination for evaluation."""
    t = _tasks.get(task_key)
    a = _algos.get(algo_key)
    if t is None:
        raise KeyError(f"Unknown task: {task_key}. Available: {list(_tasks.keys())}")
    if a is None:
        raise KeyError(f"Unknown algorithm: {algo_key}. Available: {list(_algos.keys())}")

    log_path = ROOT / "logs" / (log_subdir or _LOG_PATTERNS.get((task_key, algo_key), f"unknown/{task_key}"))
    suite = default_suite or t.default_suite

    pair = TaskAlgoPair(task=t, algo=a, log_root=log_path, default_suite=suite)
    _pairs[(task_key, algo_key)] = pair


def get_task(key: str) -> TaskDef:
    if key not in _tasks:
        raise KeyError(f"Unknown task: {key}")
    return _tasks[key]


def get_algo(key: str) -> AlgoDef:
    if key not in _algos:
        raise KeyError(f"Unknown algorithm: {key}")
    return _algos[key]


def get_pair(task_key: str, algo_key: str) -> TaskAlgoPair:
    k = (task_key, algo_key)
    if k not in _pairs:
        # Try default lookup
        register(task_key, algo_key)
    if k not in _pairs:
        raise KeyError(f"No evaluation registered for {task_key}/{algo_key}. Use register() first.")
    return _pairs[k]


def list_tasks() -> dict[str, TaskDef]:
    return dict(_tasks)


def list_algos() -> dict[str, AlgoDef]:
    return dict(_algos)


def list_pairs() -> list[TaskAlgoPair]:
    return list(_pairs.values())


# ── Built-in tasks ─────────────────────────────────────────────────────────

register_task(TaskDef(
    key="flat_walk",
    name="Flat Ground Walking",
    robot="XqRobotV2",
    default_suite="full",
    cmd_dim=5,
))

register_task(TaskDef(
    key="toe_walk",
    name="Toe Walking",
    robot="XqRobotV2",
    default_suite="toe_walk",
    cmd_dim=5,
))

register_task(TaskDef(
    key="rough_walk",
    name="Rough Terrain Walking",
    robot="XqRobotV2",
    default_suite="full",
    cmd_dim=5,
))


# ── Built-in algorithms ────────────────────────────────────────────────────

register_algo(AlgoDef(
    key="ppo",
    name="PPO",
    description="Proximal Policy Optimization (RSL-RL)",
    defaults={
        "entropy_coef": 0.002,
        "learning_rate": 1e-4,
        "clip_param": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
    },
))

register_algo(AlgoDef(
    key="sac",
    name="SAC",
    description="Soft Actor-Critic (off-policy)",
    defaults={
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
    },
))

register_algo(AlgoDef(
    key="appo",
    name="APPO",
    description="Asynchronous PPO",
    defaults={
        "entropy_coef": 0.01,
        "learning_rate": 1e-3,
        "gamma": 0.99,
    },
))

register_algo(AlgoDef(
    key="td3",
    name="TD3",
    description="Twin Delayed DDPG (off-policy)",
    defaults={
        "learning_rate": 3e-4,
        "gamma": 0.99,
    },
))


# ── Built-in task+algo pairs (trained models) ──────────────────────────────

register("flat_walk", "ppo")
register("toe_walk", "ppo")
register("rough_walk", "ppo", log_subdir="rsl_rl_ppo/XqRobotV2WalkRough")
# Future:
# register("flat_walk", "sac")
