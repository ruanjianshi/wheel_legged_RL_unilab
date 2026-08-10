"""评估输出: stdout 摘要 + JSON + Markdown 报告 + CSV (CLAUDE.md §0.1 汇报机制).

产出目录 (在 _devlog/assess/ 下, 历史 results/ 保留):
  results/<task>/<session>/metrics.json   # 机器可读指标
  reports/<task>/<session>/eval.md        # 给负责人看的评估报告
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from _devlog.assess.tasks import TaskDef, Threshold

ASSESS_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ASSESS_ROOT / "results"
REPORTS_DIR = ASSESS_ROOT / "reports"


def session_dir(task: TaskDef, run: str, ckpt: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_stem = Path(ckpt).stem
    return RESULTS_DIR / task.key / f"{run}_{ckpt_stem}_{ts}"


def _fmt(value: float | None, unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}{unit}"


def print_metrics(task: TaskDef, metrics: dict[str, float], header: str = "") -> None:
    print("=" * 60)
    print(header or f"评估 · {task.name} ({task.key})")
    print("=" * 60)
    if not metrics:
        print("  (无指标产出)")
        return
    for k, v in metrics.items():
        print(f"  {k:<20} {_fmt(v)}")


def print_verdicts(task: TaskDef, verdicts: dict[str, dict]) -> None:
    ok, passed, total = _overall_quick(verdicts)
    print("-" * 60)
    print("达标判定 (§7.x)")
    for key, v in verdicts.items():
        thr: Threshold = v["threshold"]
        mark = "✅" if v["passed"] else ("⚠️ 缺数据" if not v["ok"] else "❌")
        val = _fmt(v["value"], thr.unit)
        print(f"  {mark} {key:<18} {val:<10}  阈值 {thr.op} {thr.value}{thr.unit}")
    print(f"  总体: {passed}/{total} 达标  →  {'✅ 达标' if ok else '❌ 未达标'}")


def _overall_quick(verdicts: dict[str, dict]) -> tuple[bool, int, int]:
    total = len(verdicts)
    passed = sum(1 for v in verdicts.values() if v["passed"])
    return passed == total, passed, total


def write_json(metrics: dict[str, float], meta: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def write_markdown(
    task: TaskDef,
    metrics: dict[str, float],
    verdicts: dict[str, dict],
    meta: dict,
    out: Path,
) -> Path:
    """生成 eval.md 报告 (给负责人审阅)."""
    ok, passed, total = _overall_quick(verdicts)
    lines: list[str] = [
        f"# 评估报告 · {task.name} (`{task.key}`)",
        "",
        f"- **checkpoint**: `{meta.get('ckpt', '')}`  run: `{meta.get('run', '')}`",
        f"- **长时评估要求**: {task.long_eval or '—'}",
        f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **总体**: {'✅ 达标' if ok else '❌ 未达标'} ({passed}/{total})",
        "",
        "## 指标与达标判定 (§7.x + 附录 A)",
        "",
        "| 指标 | 值 | 阈值 | 判定 |",
        "|---|---|---|---|",
    ]
    for key, v in verdicts.items():
        thr: Threshold = v["threshold"]
        val = _fmt(v["value"], thr.unit)
        mark = "✅" if v["passed"] else ("⚠️ 缺数据" if not v["ok"] else "❌")
        lines.append(f"| `{key}` | {val} | {thr.op} {thr.value}{thr.unit} | {mark} |")
    lines.append("")
    lines.append("## 全部指标")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    for k, v in sorted(metrics.items()):
        lines.append(f"| `{k}` | {_fmt(v)} |")
    lines.append("")
    lines.append(
        "> 数据优先 (CLAUDE.md §1.5): 姿态数据 CSV 见 `logs/pose_data/`, 视频见 `video/<task>/`."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
