"""CSV and structured data export for evaluation results.

Supports:
- Flat CSV: one row per scenario, columns = metrics
- Wide CSV: one row per model, columns = scenario__metric
- Database: append to historical results DB for cross-run comparison
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def export_csv_flat(
    results: dict[str, Any],
    output_path: str | Path,
):
    """Export one row per scenario with metric columns."""
    scenario_results = results.get("results", {})
    if not scenario_results:
        return

    # Collect all metric names
    all_metrics = set()
    for sdata in scenario_results.values():
        all_metrics.update(sdata.get("metrics", {}).keys())
    metric_names = sorted(all_metrics)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            "run", "checkpoint", "suite", "scenario", "cmd_vx", "cmd_vy",
        ] + metric_names)

        for sname, sdata in scenario_results.items():
            metrics = sdata.get("metrics", {})
            writer.writerow([
                results.get("run", ""),
                results.get("checkpoint", ""),
                results.get("suite", ""),
                sname,
                sdata.get("cmd", [0, 0])[0],
                sdata.get("cmd", [0, 0])[1] if len(sdata.get("cmd", [])) > 1 else 0,
            ] + [metrics.get(m, "") for m in metric_names])


def export_csv_wide(
    results_list: list[dict[str, Any]],
    output_path: str | Path,
):
    """Export one row per model (wide format: columns = scenario__metric)."""
    if not results_list:
        return

    # Build column set
    columns = set()
    model_rows = []
    for result in results_list:
        row = {
            "run": result.get("run", ""),
            "checkpoint": result.get("checkpoint", ""),
            "suite": result.get("suite", ""),
        }
        for sname, sdata in result.get("results", {}).items():
            for mname, mval in sdata.get("metrics", {}).items():
                col = f"{sname}__{mname}"
                row[col] = mval
                columns.add(col)
        model_rows.append(row)

    sorted_cols = sorted(columns)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "checkpoint", "suite"] + sorted_cols)
        for row in model_rows:
            writer.writerow([
                row["run"], row["checkpoint"], row["suite"],
            ] + [row.get(c, "") for c in sorted_cols])


class ResultDatabase:
    """Append-only database of evaluation results for historical comparison."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _load(self) -> list[dict]:
        if self.db_path.exists():
            with open(self.db_path) as f:
                return json.load(f)
        return []

    def _save(self, data: list[dict]):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=2)

    def append(self, result: dict[str, Any]):
        data = self._load()
        entry = {
            "run": result.get("run", ""),
            "checkpoint": result.get("checkpoint", 0),
            "suite": result.get("suite", ""),
            "timestamp": result.get("evaluated_at", datetime.now().isoformat()),
            "elapsed_sec": result.get("elapsed_sec", 0),
            "results": result.get("results", {}),
        }
        data.append(entry)
        self._save(data)

    def query(self, run: str | None = None, suite: str | None = None) -> list[dict]:
        data = self._load()
        if run:
            data = [d for d in data if d["run"] == run]
        if suite:
            data = [d for d in data if d["suite"] == suite]
        return sorted(data, key=lambda d: d.get("checkpoint", 0))

    def get_trend(self, run: str, metric: str, scenario: str) -> list[tuple[int, float]]:
        """Get (checkpoint, metric_value) pairs for trend analysis."""
        entries = self.query(run=run)
        trend = []
        for e in entries:
            val = e.get("results", {}).get(scenario, {}).get("metrics", {}).get(metric)
            if val is not None:
                trend.append((e["checkpoint"], float(val)))
        return sorted(trend)
