"""渲染录制轨迹 → mp4 (复用 render_many.render_states_to_video, cam 跟踪)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def states_to_video(
    record: list[dict],
    model_path: str | Path,
    out_mp4: str | Path,
    fps: int = 50,
    num_processes: int = 1,
    cam_distance: float = 2.5,
    cam_elevation: float = -20,
    cam_azimuth: float = 135,
    cam_lookat: list[float] | None = None,
) -> Path:
    """record → state_list (每帧 [time, qpos, qvel]) → mp4."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    state_list = [np.asarray(r["state"], dtype=np.float32)[None, :] for r in record]

    from unilab.visualization.render_many import render_states_to_video

    render_states_to_video(
        state_list,
        str(model_path),
        str(out_mp4),
        fps=fps,
        width=960,
        height=540,
        num_processes=num_processes,
        cam_distance=cam_distance,
        cam_elevation=cam_elevation,
        cam_azimuth=cam_azimuth,
        cam_lookat=cam_lookat,
    )
    return out_mp4
