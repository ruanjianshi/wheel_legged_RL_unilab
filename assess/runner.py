"""XqRobotV2 Policy Assessment Runner — CLI entry point.

Categorized output structure:
    assess/
    ├── results/<task>/<algo>/<session>/    # JSON + CSV + trajectory
    ├── plots/<task>/<algo>/<session>/      # PNG charts
    ├── reports/<task>/<algo>/<session>/    # Markdown reports
    └── database/                           # Cumulative results DB

Usage:
    uv run assess/runner.py -t flat_walk -a ppo -r <run> -c <ckpt> --plot --csv --report
    uv run assess/runner.py -t flat_walk -a ppo -r <run> --trend --ckpts 5k,10k
    uv run assess/runner.py --cmp <r1.json> <r2.json> --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

from rsl_rl.modules.mlp import MLP

from assess.exporter import ResultDatabase, export_csv_flat, export_csv_wide
from assess.metrics import ALL_METRICS, METRIC_CATEGORIES, EvalContext
from assess.plotter import (
    plot_gait_phase,
    plot_metric_bars,
    plot_metric_comparison,
    plot_metric_radar,
    plot_stability_timeline,
    plot_velocity_tracking,
)
from assess.recorder import Recorder
from assess.reporter import generate_report
from assess.scenarios import SUITES, EvalScenario, EvalSuite, get_suite, load_suite
from assess.tasks import (
    AlgoDef,
    TaskAlgoPair,
    TaskDef,
    get_algo,
    get_pair,
    get_task,
    list_algos,
    list_pairs,
    list_tasks,
)
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.xqrobotV2.joystick import (
    XqRobotRewardConfig,
    XqRobotV2WalkFlatCfg,
    XqRobotV2WalkFlatEnv,
)

LOG_ROOT = ROOT / "logs" / "rsl_rl_ppo" / "XqRobotV2WalkFlat"
ASSESS_ROOT = ROOT / "assess"
RESULT_DIR = ASSESS_ROOT / "results"
PLOT_DIR = ASSESS_ROOT / "plots"
REPORT_DIR = ASSESS_ROOT / "reports"
DB_PATH = ASSESS_ROOT / "database" / "results_db.json"


# ── Session / path helpers ────────────────────────────────────────────────


class EvalSession:
    """Groups all outputs for one evaluation run under a timestamped directory."""

    def __init__(self, task: str, algo: str, run: str, ckpt: int, suite_name: str):
        self.task = task
        self.algo = algo
        self.run = run
        self.ckpt = ckpt
        self.suite = suite_name
        self.session_id = f"{run}_{ckpt}_{suite_name}_{datetime.now():%Y%m%d_%H%M%S}"

        self.result_root = RESULT_DIR / task / algo / self.session_id
        self.plot_root = PLOT_DIR / task / algo / self.session_id
        self.report_root = REPORT_DIR / task / algo / self.session_id

        for d in [self.result_root, self.plot_root, self.report_root]:
            d.mkdir(parents=True, exist_ok=True)

    def result_path(self, ext: str = "json") -> Path:
        return self.result_root / f"metrics.{ext}"

    def trajectory_path(self) -> Path:
        return self.result_root / "trajectory.json"

    def csv_path(self) -> Path:
        return self.result_root / "metrics.csv"

    def plot_path(self, name: str) -> Path:
        return self.plot_root / f"{name}.png"

    def report_path(self) -> Path:
        return self.report_root / "analysis.md"


# ── Env / Policy builders ──────────────────────────────────────────────────


def build_env_flat():
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
    return XqRobotV2WalkFlatEnv(cfg, num_envs=1, backend_type="mujoco")


_env_builders: dict[str, callable] = {}


def register_env_builder(task: str, builder: callable):
    _env_builders[task] = builder


def build_env(task: str = "flat_walk"):
    builder = _env_builders.get(task, build_env_flat)
    return builder()


register_env_builder("flat_walk", build_env_flat)


# ── Rough terrain env builder ──


def build_env_rough():
    from unilab.base.scene import TerrainSceneCfg
    from unilab.envs.locomotion.xqrobotV2.rough import (
        XqRobotRoughTerrainCfg,
        XqRobotV2WalkRoughCfg,
        XqRobotV2WalkRoughEnv,
    )

    cfg = XqRobotV2WalkRoughCfg()
    cfg.control_config.action_scale = 0.5
    cfg.control_config.wheel_action_scale = 5.0  # training default for rough
    cfg.control_config.clip_actions = 100.0
    cfg.commands = Commands(
        vel_limit=[[-1.0, -0.5, -1.5, -0.1, 0.40], [1.0, 0.5, 1.5, 0.1, 0.90]],
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
    cfg.terrain_curriculum.enabled = False
    # Use smaller terrain grid for faster single-env evaluation
    cfg.scene.terrain.generator.num_rows = 4
    cfg.scene.terrain.generator.num_cols = 4

    return XqRobotV2WalkRoughEnv(cfg, num_envs=1, backend_type="mujoco")


register_env_builder("rough_walk", build_env_rough)


def load_policy(ckpt_path: str, obs_dim: int) -> MLP:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mlp_state = {k[4:]: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")}
    mlp = MLP(obs_dim, 8, [512, 512, 256, 128], activation="elu")
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    return mlp


def find_checkpoint(run: str, ckpt: int | None = None, log_root: Path | None = None) -> Path:
    root = log_root or LOG_ROOT
    run_dir = root / run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    if ckpt is not None:
        path = run_dir / f"model_{ckpt}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    pts = sorted(
        [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not pts:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    return run_dir / pts[-1]


def find_checkpoints(run: str, ckpt_list: list[int], log_root: Path | None = None) -> list[Path]:
    root = log_root or LOG_ROOT
    run_dir = root / run
    paths = []
    for c in ckpt_list:
        p = run_dir / f"model_{c}.pt"
        if p.exists():
            paths.append(p)
    return paths


# ── Core evaluation ────────────────────────────────────────────────────────


def run_scenario(
    env, policy, scenario: EvalScenario, recorder: Recorder | None = None
) -> dict[str, float]:
    """Run single scenario, optionally recording full trajectory."""
    cmd_arr = np.array([scenario.cmd], dtype=np.float32)
    ctrl_dt = 0.01
    total_steps = int((scenario.duration + scenario.warmup) / ctrl_dt)
    skip_steps = int(scenario.warmup / ctrl_dt)

    ctx = EvalContext(cmd_vx=scenario.cmd[0], cmd_vy=scenario.cmd[1], cmd_vyaw=scenario.cmd[2])

    env._state = None
    env.init_state()
    env._state.info["commands"] = cmd_arr

    rec = None
    if recorder:
        rec = recorder.start_scenario(scenario.name, scenario.cmd)

    for s in range(total_steps):
        env._state.info["commands"] = cmd_arr

        with torch.inference_mode():
            raw_act = policy(torch.from_numpy(env._state.obs["obs"])).numpy()

        state = env.step(raw_act)
        env._state.info["commands"] = cmd_arr

        if s >= skip_steps:
            lv = env.get_local_linvel()
            gyro = env.get_gyro()
            bp = env._backend.get_base_pos()

            gravity = env._backend.get_sensor_data("upvector")
            gz = np.clip(gravity[0, 2], -1, 1)
            roll = np.arctan2(gravity[0, 1], gravity[0, 2])
            pitch = np.arcsin(np.clip(-gravity[0, 0], -1, 1))

            dof_pos = env.get_dof_pos()
            dof_vel = env.get_dof_vel()
            torque = np.zeros((1, 1))

            ctx.record(lv, gyro, bp, dof_pos, dof_vel, raw_act, torque, dof_vel[:, 6:], roll, pitch)

            if rec:
                rec.record_step(
                    t=(s - skip_steps) * ctrl_dt,
                    linvel=lv,
                    gyro=gyro,
                    base_z=bp[0, 2],
                    base_euler=np.array([roll, pitch]),
                    leg_pos=dof_pos[:, :6],
                    leg_vel=dof_vel[:, :6],
                    action=raw_act,
                    wheel_vel=dof_vel[:, 6:],
                )

    metrics = {}
    for name, fn in ALL_METRICS.items():
        try:
            metrics[name] = round(fn(ctx), 6)
        except Exception:
            metrics[name] = None

    return metrics


def evaluate(env, policy, suite: EvalSuite, record: bool = False) -> tuple[dict, Recorder | None]:
    recorder = Recorder() if record else None
    results = {}
    for scenario in suite.scenarios:
        metrics = run_scenario(env, policy, scenario, recorder)
        results[scenario.name] = {
            "description": scenario.description,
            "cmd": scenario.cmd,
            "metrics": metrics,
        }
    return results, recorder


def load_result(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Trend analysis ─────────────────────────────────────────────────────────


def run_trend(
    task: str, algo: str, run: str, ckpt_list: list[int], suite_name: str, log_root: Path
):
    suite = get_suite(suite_name)

    print(f"\nTrend: {task}/{algo} | {run} | {len(ckpt_list)} checkpoints | {suite_name}")
    env = build_env(task)
    obs_dim = env.obs_groups_spec["obs"]
    db = ResultDatabase(DB_PATH)

    results_by_ckpt: dict[int, dict] = {}

    for ckpt_iter in sorted(ckpt_list):
        ckpt_path = find_checkpoint(run, ckpt_iter, log_root=log_root)
        print(f"  [{ckpt_iter}] {ckpt_path.name}...", end=" ", flush=True)
        policy = load_policy(str(ckpt_path), obs_dim)
        t0 = time.time()
        scenario_results, _ = evaluate(env, policy, suite, record=False)
        elapsed = time.time() - t0

        session = EvalSession(task, algo, run, ckpt_iter, suite_name)
        output = {
            "run": run,
            "checkpoint": ckpt_iter,
            "suite": suite_name,
            "task": task,
            "algo": algo,
            "evaluated_at": datetime.now().isoformat(),
            "elapsed_sec": round(elapsed, 1),
            "results": scenario_results,
        }
        with open(session.result_path(), "w") as f:
            json.dump(output, f, indent=2)
        db.append(output)
        results_by_ckpt[ckpt_iter] = output
        print(f"{elapsed:.1f}s")

    env.close()

    # Trend plots
    key_metrics = [
        "vx_tracking_rmse",
        "vel_coupling",
        "vel_tracking_ratio",
        "base_height_std",
        "yaw_stability",
    ]

    for metric in key_metrics:
        fig, ax = plt.subplots(figsize=(10, 4))

        for scenario in suite.scenarios:
            x, y = [], []
            for ckpt_iter in sorted(ckpt_list):
                val = (
                    results_by_ckpt[ckpt_iter]
                    .get("results", {})
                    .get(scenario.name, {})
                    .get("metrics", {})
                    .get(metric)
                )
                if val is not None:
                    x.append(ckpt_iter)
                    y.append(val)
            if x:
                ax.plot(x, y, "-o", label=scenario.name[:20], markersize=3, lw=1.5, alpha=0.8)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} trend — {task}/{algo} / {run}")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        trend_dir = PLOT_DIR / task / algo / f"trend_{datetime.now():%Y%m%d_%H%M%S}"
        trend_dir.mkdir(parents=True, exist_ok=True)
        out = trend_dir / f"{metric}.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        print(f"  Plot: {out}")

    print("\nTrend complete.")


# ── Single evaluation (main flow) ──────────────────────────────────────────


def run_single_eval(args, pair: TaskAlgoPair, run: str, ckpt: int, suite_name: str):
    ckpt_path = find_checkpoint(run, ckpt, log_root=pair.log_root)
    suite = get_suite(suite_name)
    session = EvalSession(pair.task.key, pair.algo.key, run, ckpt, suite_name)

    print(f"Task:  {pair.display}")
    print(f"Model: {run} @ iter {ckpt}")
    print(f"Suite: {suite_name} ({len(suite.scenarios)} scenarios)")
    print(f"Output: {session.result_root}")
    if args.record:
        print("Record: enabled")

    env = build_env(pair.task.key)
    obs_dim = env.obs_groups_spec["obs"]
    policy = load_policy(str(ckpt_path), obs_dim)

    t0 = time.time()
    scenario_results, recorder = evaluate(env, policy, suite, record=args.record)
    elapsed = time.time() - t0

    output = {
        "run": run,
        "checkpoint": ckpt,
        "suite": suite_name,
        "task": pair.task.key,
        "algo": pair.algo.key,
        "evaluated_at": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "results": scenario_results,
    }

    # ── Print summary ──
    print(f"\n{'=' * 85}")
    print(f"RESULTS — {pair.display} | {run} iter={ckpt}")
    print(f"{'=' * 85}")
    header = f"{'Scenario':<22} {'vx':>7} {'vy':>7} {'vx_rmse':>8} {'vy_xtalk':>8} {'base_h':>7}"
    print(header)
    print("-" * 85)
    for sname, sdata in scenario_results.items():
        m = sdata.get("metrics", {})
        print(
            f"{sname[:21]:<22} {m.get('avg_vx', 0) or 0:>7.3f} {m.get('avg_vy', 0) or 0:>7.3f} "
            f"{m.get('vx_tracking_rmse', 0) or 0:>8.3f} {m.get('vel_coupling', 0) or 0:>8.3f} "
            f"{m.get('base_height_mean', 0) or 0:>7.3f}"
        )

    # ── Save outputs ──
    # JSON
    json_path = session.result_path()
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON: {json_path}")

    # Trajectory
    if recorder and recorder.records:
        rec_path = session.trajectory_path()
        recorder.save(rec_path)
        print(f"Traj: {rec_path}")

    # CSV
    if args.csv:
        csv_path = session.csv_path()
        export_csv_flat(output, csv_path)
        print(f"CSV:  {csv_path}")

    # Plots
    if args.plot:
        if recorder and recorder.records:
            plot_velocity_tracking(recorder.records, session.plot_path("velocity"))
            plot_stability_timeline(recorder.records, session.plot_path("stability"))
            plot_gait_phase(recorder.records, session.plot_path("gait"))
        plot_metric_bars(output.get("results", {}), session.plot_path("metric_bars"))
        print(f"Plots: {session.plot_root}/")

    # Report
    if args.report:
        report_path = session.report_path()
        generate_report(output, report_path, task=pair.task.key)
        print(f"Report: {report_path}")

    # Database
    db = ResultDatabase(DB_PATH)
    db.append(output)

    print(f"Elapsed: {elapsed:.1f}s")
    env.close()


# ── Comparison ─────────────────────────────────────────────────────────────


def run_comparison(paths: list[str], args):
    names = [Path(p).stem[:40] for p in paths]
    data_list = [load_result(p) for p in paths]

    print(f"\nComparing {len(paths)} results:")
    for name in names:
        print(f"  {name}")

    print(f"\n{'Scenario':<22} | ", end="")
    for name in names:
        print(f"{name:<15} | ", end="")
    print()

    all_scenarios = set()
    for d in data_list:
        all_scenarios.update(d.get("results", {}).keys())

    for scenario in sorted(all_scenarios):
        print(f"{scenario[:21]:<22} | ", end="")
        for d in data_list:
            m = d.get("results", {}).get(scenario, {}).get("metrics", {})
            vx_rmse = m.get("vx_tracking_rmse", "N/A")
            vy_xt = m.get("vel_coupling", "N/A")
            if isinstance(vx_rmse, float):
                print(f"vx={vx_rmse:.3f} vy={vy_xt:.3f} | ", end="")
            else:
                print(f"{'N/A':<15} | ", end="")
        print()

    if args.plot:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_dir = PLOT_DIR / f"comparison_{ts}"
        comp_dir.mkdir(parents=True, exist_ok=True)

        comparison_data = {}
        for i, d in enumerate(data_list):
            agg: dict[str, list[float]] = {}
            for sdata in d.get("results", {}).values():
                for mk, mv in sdata.get("metrics", {}).items():
                    if mv is not None:
                        agg.setdefault(mk, []).append(float(mv))
            comparison_data[names[i]] = {k: np.mean(v) for k, v in agg.items()}

        plot_metric_radar(comparison_data, comp_dir / "radar.png")
        comps = [
            {"label": names[i], "results": d.get("results", {})} for i, d in enumerate(data_list)
        ]
        for metric in ["vx_tracking_rmse", "vel_coupling", "vel_tracking_ratio"]:
            plot_metric_comparison(comps, comp_dir / f"comparison_{metric}.png", metric_key=metric)
        print(f"Plots: {comp_dir}/")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="XqRobotV2 Policy Assessor")
    parser.add_argument("--task", "-t", default="flat_walk", help="Task key: flat_walk, toe_walk")
    parser.add_argument("--algo", "-a", default="ppo", help="Algorithm: ppo, sac, appo, td3")
    parser.add_argument("--run", "-r", help="Training run name")
    parser.add_argument("--ckpt", "-c", type=int, help="Checkpoint iteration")
    parser.add_argument("--ckpts", help="Comma-separated checkpoints for trend")
    parser.add_argument("--suite", "-s", help="Suite name (default: task default)")
    parser.add_argument("--suite-file", help="Custom YAML suite file")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--list-suites", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--cmp", "--compare", nargs="+")
    parser.add_argument("--trend", action="store_true")
    args = parser.parse_args()

    if args.list_tasks:
        print("Registered algorithms:")
        for key, a in list_algos().items():
            print(f"  {key:8s} {a.name} — {a.description}")
        print("\nRegistered task+algorithm pairs:")
        for p in list_pairs():
            print(f"  {p.key:20s} → {p.log_root}")
        return

    if args.list_suites:
        print("Available suites:")
        for name, suite in SUITES.items():
            print(f"  {name:15s} {suite.description} ({len(suite.scenarios)} scenarios)")
        return

    if args.cmp:
        run_comparison(args.cmp, args)
        return

    if args.trend:
        if not args.run or not args.ckpts:
            parser.error("--run and --ckpts required for --trend")
        pair = get_pair(args.task, args.algo)
        ckpt_list = [int(x.strip()) for x in args.ckpts.split(",")]
        suite_name = args.suite or pair.default_suite
        run_trend(args.task, args.algo, args.run, ckpt_list, suite_name, pair.log_root)
        return

    if not args.run:
        parser.error("--run required")

    pair = get_pair(args.task, args.algo)
    ckpt_iter = args.ckpt
    ckpt_path = find_checkpoint(args.run, ckpt_iter, log_root=pair.log_root)
    resolved_ckpt = ckpt_iter or int(ckpt_path.stem.split("_")[1])
    suite_name = args.suite or pair.default_suite

    run_single_eval(args, pair, args.run, resolved_ckpt, suite_name)


if __name__ == "__main__":
    main()
