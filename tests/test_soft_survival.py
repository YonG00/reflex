import torch
from reflex.world_model.events_v24.soft_survival import expected_terminal_reward
from reflex.world_model.joint import returns


def test_expected_terminal_cost_is_first_event_probability_mass():
    h = 32
    z = torch.tensor(-1.0, requires_grad=True)
    p = z.sigmoid().expand(h, 1)
    rewards = expected_terminal_reward(torch.zeros(h, 1, 3), p, 1, -2.0)
    imagined = {
        "reward": rewards.sum(-1),
        "done": p,
        "obs": torch.zeros(h + 1, 1, 1),
        "truncated": torch.zeros(h, 1, dtype=torch.bool),
    }
    value = returns(imagined, lambda x: torch.zeros(*x.shape[:-1], 1), 1.0)
    expected = -2 * (1 - (1 - z.sigmoid()) ** h)
    torch.testing.assert_close(value.squeeze(), expected)
    # Increasing hazard increases expected terminal cost; it cannot escape
    # repeated negative fees by terminating sooner.
    (-value.mean()).backward()
    assert z.grad > 0
    assert rewards[..., 0].eq(0).all() and rewards[..., 2].eq(0).all()
