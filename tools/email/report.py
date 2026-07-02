"""自循环最终报告生成 + 邮件发送 CLI

用法:
    # 指定评估结果路径发送
    uv run tools/email/report.py \
        -t flat_walk -a ppo \
        -r 2026-07-01_13-55-35_mujoco -c 20000 \
        --to qfantastic@2925.com

    # 指定 SMTP 凭据 (或设环境变量 UNILAB_SMTP_USER/UNILAB_SMTP_PASS)
    SMTP_HOST=smtp.2925.com SMTP_PORT=465 \
    SMTP_USER=x15347348975 SMTP_PASS=x15347348975 \
    uv run tools/email/report.py -t flat_walk -a ppo -r <run> -c <iter> --to qfantastic@2925.com

    # 预览报告(不发送)
    uv run tools/email/report.py -t flat_walk -a ppo -r <run> -c <iter> --preview
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSESS_DIR = PROJECT_ROOT / "assess"
LOG_ROOT = PROJECT_ROOT / "logs"


def find_latest_assess_result(task: str, algo: str, run: str, ckpt: int) -> Path | None:
    results_root = ASSESS_DIR / "results" / task / algo
    if not results_root.exists():
        return None

    prefix = f"{run}_{ckpt}_"
    dirs = sorted(
        [d for d in results_root.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        reverse=True,
    )
    if not dirs:
        return None
    return dirs[0]


def load_metrics(metrics_json: Path) -> dict:
    with open(metrics_json) as f:
        return json.load(f)


def load_train_log(run: str) -> dict:
    """从 run_config.json 读取训练参数"""
    run_dir = LOG_ROOT / "rsl_rl_ppo" / "XqRobotV2WalkFlat" / run
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        # 尝试 rough
        run_dir = LOG_ROOT / "rsl_rl_ppo" / "XqRobotV2WalkRough" / run
        cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def compute_summary(results: dict) -> dict:
    """从分场景结果计算汇总指标"""
    vx_rmse_list = []
    vy_rmse_list = []
    crosstalk_list = []
    base_h_list = []
    for name, sc in results.items():
        m = sc.get("metrics", {})
        if "vx_tracking_rmse" in m:
            vx_rmse_list.append(m["vx_tracking_rmse"])
        if "vy_tracking_rmse" in m:
            vy_rmse_list.append(m["vy_tracking_rmse"])
        if "vel_coupling" in m:
            crosstalk_list.append(abs(m["vel_coupling"]))
        if "base_height_mean" in m:
            base_h_list.append(m["base_height_mean"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    return {
        "vx_rmse_avg": avg(vx_rmse_list),
        "vy_rmse_avg": avg(vy_rmse_list),
        "vy_crosstalk_avg": avg(crosstalk_list),
        "base_height_mean": avg(base_h_list),
    }


def build_report(
    task: str,
    algo: str,
    run: str,
    ckpt: int,
) -> str:
    task_name_map = {
        "flat_walk": "平坦行走",
        "rough_walk": "粗糙地形行走",
        "toe_walk": "脚趾行走",
    }
    task_cn = task_name_map.get(task, task)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 加载评估数据
    result_dir = find_latest_assess_result(task, algo, run, ckpt)
    metrics = {}
    if result_dir:
        metrics_json = result_dir / "metrics.json"
        if metrics_json.exists():
            metrics = load_metrics(metrics_json)

    # 加载训练配置
    train_cfg = load_train_log(run)
    hyper = {}
    if train_cfg:
        hyper["num_envs"] = train_cfg.get("num_envs", "?")
        hyper["max_iterations"] = train_cfg.get("max_iterations", "?")

    lines = []
    lines.append("=" * 64)
    lines.append(f"  UniLab XqRobotV2 RL 训练自循环报告")
    lines.append("=" * 64)
    lines.append(f"")
    lines.append(f"  任务: {task_cn} ({task})")
    lines.append(f"  算法: {algo.upper()}")
    lines.append(f"  训练: {run} | iter {ckpt}")
    lines.append(f"  时间: {now}")
    lines.append(f"")

    if hyper:
        lines.append(f"[训练参数]")
        lines.append(f"  num_envs={hyper.get('num_envs')}, max_iter={hyper.get('max_iterations')}")
        lines.append(f"")

    # 分场景
    results = metrics.get("results", {})
    if results:
        summary = compute_summary(results)

        lines.append(f"[综合指标]")
        lines.append(f"  Vx 跟踪 RMSE avg:          {summary.get('vx_rmse_avg', 'N/A'):.4f}")
        lines.append(f"  Vy 跟踪 RMSE avg:          {summary.get('vy_rmse_avg', 'N/A'):.4f}")
        lines.append(f"  VxVy 串扰 (crosstalk) avg: {summary.get('vy_crosstalk_avg', 'N/A'):.4f}")
        lines.append(f"  Base 高度 mean:            {summary.get('base_height_mean', 'N/A'):.4f}")
        lines.append(f"")

        lines.append(f"[分场景结果]")
        lines.append(f"  {'场景':<24s} {'Vx':>7s} {'Vy':>7s} {'VxRMSE':>7s} {'VyXtalk':>8s} {'BaseH':>7s}")
        lines.append(f"  {'-'*60}")
        for name, sc in results.items():
            m = sc.get("metrics", {})
            lines.append(
                f"  {name:<24s} {m.get('avg_vx', 0):>7.3f} {m.get('avg_vy', 0):>7.3f} "
                f"{m.get('vx_tracking_rmse', 0):>7.3f} {abs(m.get('vel_coupling', 0)):>8.3f} "
                f"{m.get('base_height_mean', 0):>7.3f}"
            )
        lines.append(f"")

        # 自动结论
        lines.append(f"[自动结论]")
        conclusions = []
        vx_rmse = summary.get("vx_rmse_avg", 1.0)
        vy_xtalk = summary.get("vy_crosstalk_avg", 1.0)
        base_h = summary.get("base_height_mean", 0)

        if vx_rmse < 0.15:
            conclusions.append(f"✅ Vx 跟踪良好 (RMSE={vx_rmse:.4f} < 0.15)")
        elif vx_rmse < 0.30:
            conclusions.append(f"⚠️  Vx 跟踪一般 (RMSE={vx_rmse:.4f}) — 需继续训练")
        else:
            conclusions.append(f"❌ Vx 跟踪差 (RMSE={vx_rmse:.4f}) — 需调整超参")

        if vy_xtalk < 0.10:
            conclusions.append(f"✅ Vy 串扰小 ({vy_xtalk:.4f} < 0.10) — Vx/Vy 解耦良好")
        elif vy_xtalk < 0.30:
            conclusions.append(f"⚠️  Vy 串扰中等 ({vy_xtalk:.4f})")
        else:
            conclusions.append(f"❌ Vy 串扰大 ({vy_xtalk:.4f}) — 检查髋对称性")

        if 0.60 < base_h < 0.70:
            conclusions.append(f"✅ 高度正常 (mean={base_h:.4f})")
        else:
            conclusions.append(f"⚠️  高度偏离 (mean={base_h:.4f}, target=0.65)")

        for c in conclusions:
            lines.append(f"  {c}")
    else:
        lines.append(f"  (无评估数据，请先运行 assess)")

    lines.append(f"")
    lines.append(f"[评估数据路径]")
    if result_dir:
        lines.append(f"  {result_dir}")
    else:
        lines.append(f"  (未找到评估数据)")
    lines.append(f"")
    lines.append(f"=" * 64)
    lines.append(f"  由 UniLab AI 自循环系统自动生成")
    lines.append(f"=" * 64)

    return "\n".join(lines)


def build_html_report(text: str) -> str:
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }}
  pre {{ background: #161b22; padding: 16px; border-radius: 6px; white-space: pre-wrap; }}
  h2 {{ color: #58a6ff; }}
  .ok {{ color: #3fb950; }}
  .warn {{ color: #d29922; }}
  .fail {{ color: #f85149; }}
</style></head><body>
<pre>{text}</pre>
</body></html>"""
    return html


def main():
    parser = argparse.ArgumentParser(description="自循环最终报告生成 & 邮件发送")
    parser.add_argument("-t", "--task", default="flat_walk")
    parser.add_argument("-a", "--algo", default="ppo")
    parser.add_argument("-r", "--run", required=True, help="训练 run ID")
    parser.add_argument("-c", "--ckpt", type=int, required=True, help="checkpoint iter")
    parser.add_argument("--to", default="qfantastic@2925.com", help="收件人")
    parser.add_argument("--preview", action="store_true", help="仅预览报告，不发邮件")
    parser.add_argument("--smtp-host", default=os.environ.get("UNILAB_SMTP_HOST", "smtp.2925.com"))
    parser.add_argument("--smtp-port", type=int, default=int(os.environ.get("UNILAB_SMTP_PORT", "465")))
    parser.add_argument("--smtp-user", default=os.environ.get("UNILAB_SMTP_USER", ""))
    parser.add_argument("--smtp-pass", default=os.environ.get("UNILAB_SMTP_PASS", ""))
    args = parser.parse_args()

    text = build_report(args.task, args.algo, args.run, args.ckpt)
    print(text)

    if args.preview:
        return

    # SMTP 凭据: CLI 参数 > 环境变量 > 默认
    sys.path.insert(0, str(PROJECT_ROOT))
    from tools.email.config import SmtpConfig
    from tools.email.sender import send_email

    cfg = SmtpConfig(
        host=args.smtp_host,
        port=args.smtp_port,
        user=args.smtp_user,
        password=args.smtp_pass,
    )

    if not cfg.user or not cfg.password:
        print("\n[email] 未配置 SMTP 凭据。请设置环境变量或传参 --smtp-user/--smtp-pass")
        sys.exit(1)

    now = datetime.now().strftime("%m-%d %H:%M")
    subject = f"[UniLab] {args.task}/{args.algo} @ iter {args.ckpt} — {now}"
    ok = send_email(
        to=args.to,
        subject=subject,
        body=text,
        body_html=build_html_report(text),
        cfg=cfg,
    )
    if ok:
        print(f"\n[email] 已发送 → {args.to}")
    else:
        print(f"\n[email] 发送失败")
        sys.exit(1)


if __name__ == "__main__":
    main()