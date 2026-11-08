import pytest
import torch
from reflex.world_model.models import Standardizer
from reflex.world_model.native_pointwise_v35 import norm_native, silu_native


@pytest.mark.parametrize("compiled", [False, True])
def test_native_silu_values_and_gradient(compiled):
    values = torch.tensor(
        [-30.0, -2.0, -1e-7, 0.0, 1e-7, 2.0, 30.0], requires_grad=True
    )
    call = (
        torch.compile(silu_native, fullgraph=True, backend="aot_eager")
        if compiled
        else silu_native
    )
    expected = torch.nn.functional.silu(values)
    actual = call(values)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    a = torch.autograd.grad(actual.square().sum(), values)[0]
    b = torch.autograd.grad(expected.square().sum(), values)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("compiled", [False, True])
def test_native_norm_clamp_boundaries_and_gradient(compiled):
    model = Standardizer(7)
    model.mean.copy_(torch.arange(7.0) / 10)
    model.std.copy_(torch.arange(7.0) / 10 + 0.01)
    value = (
        model.mean
        + model.std * torch.tensor([-21.0, -20.0, -1e-6, 0.0, 1e-6, 20.0, 21.0])
    ).requires_grad_()
    call = (
        torch.compile(norm_native, fullgraph=True, backend="aot_eager")
        if compiled
        else norm_native
    )
    expected = model(value)
    actual = call(value, model.mean, model.std)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    a = torch.autograd.grad(actual.sum(), value)[0]
    b = torch.autograd.grad(expected.sum(), value)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_trainable_statistics_are_rejected():
    with pytest.raises(ValueError, match="fixed statistics"):
        norm_native(torch.zeros(2), torch.zeros(2, requires_grad=True), torch.ones(2))


def test_real_replay_h8_and_full_actor_check_restores_state(tmp_path):
    from reflex.world_model.diagnostics_v31.state import digest
    from reflex.world_model.physical_full_v35 import check
    from test_actor_diagnostics_v31 import fixture

    t = fixture(tmp_path)
    t.cfg.model_window = 8
    t.cfg.compile_backend = "aot_eager"
    model = digest(t.model.state_dict())
    actor = digest(t.actor.state_dict())
    rng = torch.get_rng_state().clone()
    result = check(t, tmp_path / "full", repeats=1, warmups=1)
    assert result["completed"] and result["candidate_numeric_passed"], result
    assert (
        digest(t.model.state_dict()) == model and digest(t.actor.state_dict()) == actor
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert not hasattr(t.model, "_orig_mod")
    assert "predict_physical" not in t.model.__dict__
