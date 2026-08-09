from __future__ import annotations

import pytest
import torch
from benchmark import benchmark_replay_buffer_placement as bench


def test_replay_shape_packed_width_includes_critic_fields() -> None:
    shape = bench.ReplayShape(obs_dim=45, action_dim=29, critic_dim=48)

    assert shape.packed_width == 2 * 45 + 29 + 3 + 2 * 48


def test_td3_is_an_allowed_benchmark_algo() -> None:
    assert "td3" in bench._parse_algos("sac,flashsac,td3")


def test_parse_tasks_deduplicates_ordered_values() -> None:
    assert bench._parse_tasks("g1_walk_flat,g1_motion_tracking,g1_walk_flat") == [
        "g1_walk_flat",
        "g1_motion_tracking",
    ]


def test_parse_tasks_auto_must_stand_alone() -> None:
    assert bench._parse_tasks("auto") == ["auto"]
    with pytest.raises(ValueError, match="cannot be combined"):
        bench._parse_tasks("auto,g1_walk_flat")


def test_resolve_device_auto_prefers_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        bench.torch, "xpu", type("_Xpu", (), {"is_available": lambda: False}), raising=False
    )
    monkeypatch.setattr(bench.torch.backends.mps, "is_available", lambda: True)

    assert bench._resolve_device("auto") == torch.device("mps")


def test_resolve_device_auto_prefers_xpu_before_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        bench.torch, "xpu", type("_Xpu", (), {"is_available": lambda: True}), raising=False
    )
    monkeypatch.setattr(bench.torch.backends.mps, "is_available", lambda: True)

    assert bench._resolve_device("auto") == torch.device("xpu")


def test_resolve_device_rejects_unavailable_xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bench.torch, "xpu", type("_Xpu", (), {"is_available": lambda: False}), raising=False
    )

    with pytest.raises(ValueError, match="XPU was requested"):
        bench._resolve_device("xpu")


def test_resolve_device_rejects_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="MPS was requested"):
        bench._resolve_device("mps")


def test_replay_transfer_manifest_records_backend_fields() -> None:
    manifest = bench._replay_transfer_manifest(torch.device("cpu"))

    assert manifest["backend"] == "TorchCopyReplayTransferBackend"
    assert manifest["device_family"] == "cpu"
    assert manifest["host_memory_kind"] == "pageable_shared"
    assert manifest["supports_async_submit"] is False
    assert manifest["ring_depth"] == 2


def test_run_case_cpu_portable_path_records_device_transfer_timings() -> None:
    case = bench.BenchmarkCase(
        algo="sac",
        task="dummy",
        sim="mujoco",
        command="uv run train --algo sac --task dummy --sim mujoco",
        training_task_name="Dummy",
        num_envs=2,
        env_steps_per_sync=1,
        replay_buffer_n=4,
        config_capacity_rows=8,
        benchmark_capacity_rows=8,
        configured_batch_size=2,
        learner_batch_size=2,
        symmetry_batch_multiplier=1,
        updates_per_step=1,
        sample_count=2,
        learning_starts=0,
        incremental_rows=2,
        shape=bench.ReplayShape(obs_dim=3, action_dim=2, critic_dim=1),
    )

    result = bench._run_case(
        case,
        device=torch.device("cpu"),
        warmup=0,
        repeat=1,
        prefill="none",
        incremental_source_pinned=False,
        sampled_batch_pinned=False,
    )

    assert "gpu_full_random_sample" in result.timings
    assert "current_ipc_incremental_h2d" in result.timings
    assert "cpu_full_presample" in result.timings
    assert "cpu_sampled_batch_h2d" in result.timings
    assert result.notes == []
