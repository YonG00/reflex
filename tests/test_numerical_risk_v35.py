import math

import torch
from reflex.world_model.numerical_risk_v35 import (
    actor_risk,
    physical_summary,
    vector_summary,
)


def test_physical_units_and_sign_invariant_small_angles():
    a = torch.zeros(2, 84)
    a[:, 3] = 1
    a[:, 74] = 1
    b = a.clone()
    b[:, 71] = 1e-4
    b[:, 82] = 2e-4
    b[:, 74:78] *= -1
    report = physical_summary(a, b)
    assert abs(report["box_position_m"]["max_abs"] - 1e-4) < 1e-10
    assert report["box_orientation_error_rad"]["max"] == 0
    angle = 1e-4
    b[:, 3] = math.cos(angle / 2)
    b[:, 6] = math.sin(angle / 2)
    report = physical_summary(a, b)
    assert abs(report["root_orientation_error_rad"]["max"] - angle) < 1e-8


def test_update_direction_compares_deltas_and_keeps_small_gradients():
    ref = {"gradient/0": torch.tensor([2.65e-9]), "parameter/x": torch.tensor([1.001])}
    new = {"gradient/0": torch.tensor([1.13e-9]), "parameter/x": torch.tensor([0.999])}
    report = actor_risk(ref, new, {"x": torch.tensor([1.0])}, [("x", None)])
    assert report["actual_adam_delta"]["cosine"] == -1
    assert report["applied_gradient"]["relative_l2"] > 0.5
    assert "passed" not in report


def test_invalid_and_zero_vectors_do_not_claim_valid_direction():
    assert not vector_summary(torch.zeros(2), torch.ones(2))["direction_defined"]
    assert not vector_summary(torch.ones(2), torch.tensor([float("nan"), 0]))["finite"]
    a = torch.zeros(1, 84)
    assert not physical_summary(a, a)["root_orientation_error_rad"]["valid_quaternions"]


def test_survival_weighted_diagnostic_does_not_hide_unweighted_error():
    a = torch.zeros(2, 1, 84)
    a[..., 3] = 1
    a[..., 74] = 1
    b = a.clone()
    b[1, 0, 0] = 1
    report = physical_summary(a, b, weights=torch.tensor([[1.0], [0.0]]))
    assert report["root_position_m"]["max_abs"] == 1
    assert report["root_position_m"]["reference_survival_weighted_rms"] == 0
    assert "passed" not in report["root_position_m"]


def test_full_trajectory_risk_keeps_event_checks_and_weighted_error():
    from reflex.world_model.numerical_risk_v35 import trajectory_risk

    states = torch.zeros(2, 1, 84)
    states[..., 3] = 1
    states[..., 74] = 1
    ref = {
        "actor_gradient": torch.ones(2),
        "physical_states": states,
        "reward": torch.ones(2, 1),
        "style": torch.ones(2, 1),
        "done": torch.full((2, 1), 0.2),
        "valid": torch.tensor([[True], [False]]),
        "truncated": torch.zeros(2, 1, dtype=torch.bool),
        "predicted_launch_probability": torch.zeros(2, 1),
        "event_trace": torch.zeros(2, 1, 340),
    }
    actual = {k: v.clone() for k, v in ref.items()}
    actual["physical_states"][1, 0, 0] = 1
    report = trajectory_risk(ref, actual)
    assert report["finite_all"]
    assert report["event_write_mask_disagreements"] == 0
    assert report["physical_state"]["root_position_m"]["max_abs"] == 1
    assert (
        report["physical_state"]["root_position_m"]["reference_survival_weighted_rms"]
        == 0
    )
