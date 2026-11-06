"""Numeric pre-write/post-write transitions, independent of launched labels."""

import torch
from reflex.world_model.events import physical_write_mask
from reflex.world_model.replay import Replay


def test_velocity_write_is_not_a_physics_target_or_a_continuous_window():
    replay = Replay(12)
    state = torch.zeros(6, 84)
    state[:, 3] = state[:, 74] = 1
    for j in range(12):
        physical = state.clone()
        physical[:, 78] *= 0.9  # A real continuous damping step.
        post_control = physical.clone()
        if j == 1:
            post_control[:, 78] = 4  # Native launch-like write, flag stays TRUE.
        boundary = physical_write_mask(physical, post_control)
        replay.add(
            {
                "state": state,
                "next_state": physical,
                "valid": torch.ones(6, dtype=torch.bool),
                "boundary": boundary,
                "terminated": torch.zeros(6, dtype=torch.bool),
                "timeout": torch.zeros(6, dtype=torch.bool),
                "launched": torch.ones(6, dtype=torch.bool),
                "phase": torch.full((6,), j),
                "reference_length": torch.full((6,), 100),
            }
        )
        state = post_control
    assert replay.ordered("next_state")[1, 1, 78] == 0
    assert replay.ordered("state")[2, 1, 78] == 4
    assert replay.ordered("boundary")[1].all()
    assert not replay.ordered("boundary")[2:].any()
    ix = replay.indices(8)
    assert (ix[:, 0] >= 2).all()
    assert (ix[:, 0] == 2).any()
    # Post-write states and physical deceleration remain supervised.
    sequence = replay.sample(64, 8)
    assert (sequence["state"][0, :, 78] > 0).all()
    assert (sequence["next_state"][:, :, 78] < sequence["state"][:, :, 78]).all()


def test_hover_write_keeps_robot_physics_and_excludes_artificial_box_restore():
    replay = Replay(3)
    for j in range(3):
        start = torch.zeros(6, 84)
        physical = start.clone()
        physical[:, 0] = 0.02 * (j + 1)  # Robot's real physical displacement.
        physical[:, 80] = -0.1962  # Gravity during the physical step.
        held = physical.clone()
        held[:, 80] = 0  # Program restores the hovering box after reward capture.
        replay.add(
            {
                "state": start,
                "next_state": physical,
                "valid": torch.ones(6, dtype=torch.bool),
                "boundary": physical_write_mask(physical, held),
                "terminated": torch.zeros(6, dtype=torch.bool),
                "timeout": torch.zeros(6, dtype=torch.bool),
                "launched": torch.zeros(6, dtype=torch.bool),
                "phase": torch.full((6,), j),
                "reference_length": torch.full((6,), 100),
            }
        )
    assert not len(replay.indices(8))
    batch = replay.sample_variable_windows(32, 8)
    assert batch["loss_mask"].sum(0).eq(1).all()
    assert (batch["next_state"][0, :, 0] > 0).all()
    assert torch.allclose(batch["next_state"][0, :, 80], torch.full((32,), -0.1962))
    assert not physical_write_mask(held, held.clone()).any()


def test_macro_control_writes_are_targets_but_reference_reset_is_not():
    from reflex.world_model.events import macro_step_target

    physical = torch.zeros(4, 84)
    post = physical.clone()
    post[:, 78] = 4
    post[2:, 0] = 100  # Random reset/root teleport.
    done = torch.tensor([False, False, True, False])
    reference_jump = torch.tensor([False, False, False, True])
    before_mode = torch.zeros(4, dtype=torch.bool)
    after_mode = torch.ones(4, dtype=torch.bool)
    target, mode, reset = macro_step_target(
        physical, post, done, reference_jump, before_mode, after_mode
    )
    assert target[:2, 78].eq(4).all() and mode[:2].all()
    torch.testing.assert_close(target[2:], physical[2:])
    assert not mode[2:].any()
    torch.testing.assert_close(reset, torch.tensor([False, False, True, True]))
