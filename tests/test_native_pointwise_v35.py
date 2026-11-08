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
