"""Regression tests for standing-to-single-leg command semantics."""

from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.xqrobotwl.single_leg import (
    _compute_feedforward,
    _reward_balance_complete,
    _reward_fold_pose,
    _reward_stance_height,
    _update_fsm,
)


def test_transition_feedforward_moves_free_leg_counterweight() -> None:
    early = _compute_feedforward(np.array([0]), np.array([0.0]), action_scale=0.6)
    late = _compute_feedforward(np.array([0]), np.array([1.0]), action_scale=0.6)
    hold = _compute_feedforward(np.array([1]), np.array([0.0]), action_scale=0.6)
    assert early[0, 0] == 0.0
    assert late[0, 0] < -0.9
    assert hold[0, 0] == late[0, 0]


def test_balance_state_is_held_until_trigger_is_released() -> None:
    state = np.array([1], dtype=np.int32)
    timer = np.array([0.8])

    state, timer = _update_fsm(
        state,
        timer,
        sl_trigger=np.array([1.0]),
        balance_done=np.array([True]),
        dt=0.01,
    )
    assert state.tolist() == [1]

    state, timer = _update_fsm(
        state,
        timer,
        sl_trigger=np.array([0.0]),
        balance_done=np.array([True]),
        dt=0.01,
    )
    assert state.tolist() == [2]
    assert timer.tolist() == [0.0]


def test_balance_completion_reward_is_a_one_step_event() -> None:
    ctx = RewardContext(
        info={
            "balance_completed": np.array([True, True]),
            "balance_just_completed": np.array([True, False]),
        },
        linvel=np.zeros((2, 3)),
        gyro=np.zeros((2, 3)),
        dof_pos=np.zeros((2, 6)),
        num_envs=2,
    )
    assert _reward_balance_complete(ctx).tolist() == [1.0, 0.0]


def test_pose_and_height_helpers_return_positive_costs() -> None:
    ctx = RewardContext(
        info={"fsm_state": np.array([1])},
        linvel=np.zeros((1, 3)),
        gyro=np.zeros((1, 3)),
        dof_pos=np.ones((1, 6)),
        num_envs=1,
        base_height=np.array([0.3]),
        base_height_target=0.55,
    )
    assert _reward_fold_pose(ctx)[0] > 0.0
    assert _reward_stance_height(ctx)[0] > 0.0
