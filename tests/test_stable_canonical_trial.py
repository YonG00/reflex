"""A proven unit-quaternion identity, opt-in and benchmark-only."""

import pytest
import torch
from reflex.world_model.physics import canonical, expmap, mul, normalize
from reflex.world_model.stable_canonical import trial
from test_analytic_soft_termination import new_trainer
from test_compact_context import compact_trainer, fill, robot_state


def test_deyaw_r10_zero_and_zero_tangent_gradient_float64():
    torch.manual_seed(308)
    q = normalize(torch.randn(128, 4, dtype=torch.float64))
    tangent = (torch.randn(128, 3, dtype=torch.float64) * 0.1).requires_grad_()
    rotated = normalize(mul(expmap(tangent), q))
    base = torch.zeros(128, 84, dtype=torch.float64)
    base[:, 74] = 1
    state = torch.cat((base[:, :3], rotated, base[:, 7:]), -1)
    value = canonical(state)[:, 3]
    assert value.abs().max() < 1e-13
    gradient = torch.autograd.grad(value.sum(), tangent)[0]
    assert gradient.abs().max() < 1e-12


def test_stable_features_only_zero_theoretical_column_and_restore_on_error(tmp_path):
    trainer, row = compact_trainer(tmp_path)
    state = robot_state(4)
    target, event = torch.zeros(4, 29), torch.zeros(4, 4)
    original = trainer.model.features(state, target, row["context"], event)
    with pytest.raises(RuntimeError, match="probe"), trial(trainer):
        actual = trainer.model.features(state, target, row["context"], event)
        assert actual[:, 3].eq(0).all()
        torch.testing.assert_close(actual[:, :3], original[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(actual[:, 4:], original[:, 4:], atol=0, rtol=0)
        raise RuntimeError("probe")
    assert "features" not in trainer.model.__dict__
    assert not hasattr(trainer.model, "_stable_heading_trial")
    torch.testing.assert_close(
        trainer.model.features(state, target, row["context"], event),
        original,
        atol=0,
        rtol=0,
    )


def test_stable_benchmark_restores_original_encoder_and_weights(tmp_path):
    from reflex.world_model.wm_rollout_benchmark import check

    source, row = compact_trainer(tmp_path)
    trainer = new_trainer(source, compile_backend="aot_eager")
    fill(trainer, row)
    trainer.cfg.model_window, trainer.cfg.model_steps = 2, 1
    initial = {k: v.clone() for k, v in trainer.model.net.state_dict().items()}
    result = check(trainer, tmp_path, repeats=1, stable_heading=True)
    assert result["completed"] and result["numeric_passed"]
    assert result["stable_heading_trial"]
    assert "features" not in trainer.model.__dict__
    for key, value in trainer.model.net.state_dict().items():
        torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
    torch._dynamo.reset()
