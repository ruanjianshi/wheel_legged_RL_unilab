from __future__ import annotations

from pathlib import Path

from unilab.utils.support_matrix import EvidenceLevel, build_support_rows


def _row(entrypoint_label: str, task_slug: str):
    root = Path(__file__).resolve().parents[2]
    for row in build_support_rows(root):
        if row.entrypoint_label == entrypoint_label and row.task_slug == task_slug:
            return row
    raise AssertionError(f"Missing support row: {entrypoint_label} / {task_slug}")


def test_support_matrix_marks_xqrobotV2_ppo_backends_as_tested():
    row = _row("PPO (torch)", "xqrobotV2_walk_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED
    assert row.cells["motrix"].level == EvidenceLevel.TESTED


def test_support_matrix_marks_xqrobotwl_mujoco_backends_as_tested():
    row = _row("PPO (torch)", "xqrobotwl_walk_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.TESTED


def test_support_matrix_keeps_uncovered_mlx_tasks_at_configured():
    row = _row("PPO (mlx)", "xqrobotwl_jump_flat")

    assert row.cells["mujoco"].level == EvidenceLevel.CONFIGURED
