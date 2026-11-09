import pytest
import torch
from reflex.world_model.task import HISTORY_FIELDS
from test_analytic_task import batch, make_task, state


def test_diagnostic_geometry_is_explicit_and_never_enters_learning():
    task = make_task()
    b = batch()
    s = state()
    action = torch.zeros(2, 29)
    phase = torch.zeros(2, dtype=torch.long)
    history = torch.zeros(2, len(HISTORY_FIELDS))
    steps = torch.zeros(2, dtype=torch.long)
    seconds = torch.ones(2)
    args = (b, s, s, action, action, phase, history, steps, seconds)
    expected = task.evaluate_step(*args)
    view = task._view(b, s, s, action, action, phase, history, steps + 1, 0)
    with pytest.raises(ValueError, match="restricted"):
        task.evaluate_step(*args, diagnostic_view=view)
    view._native_parity_only = True
    actual = task.evaluate_step(*args, diagnostic_view=view)
    torch.testing.assert_close(expected.rewards, actual.rewards, rtol=0, atol=0)
    torch.testing.assert_close(expected.terminated, actual.terminated)
    bad = (b, s, s, action.requires_grad_(), action, phase, history, steps, seconds)
    with pytest.raises(ValueError, match="learning graph"):
        task.evaluate_step(*bad, diagnostic_view=view)
    # Unchanged compact-state route still propagates Actor/action gradients.
    value = task.evaluate_step(*bad)
    assert torch.autograd.grad(value.rewards.sum(), action)[0].isfinite().all()
