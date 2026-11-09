import copy

import torch
from reflex.world_model.critic_values import evaluate
from reflex.world_model.joint import returns


def test_checkpoint_chunks_preserve_soft_survival_truncation_gradients():
    torch.manual_seed(52)
    critic = torch.nn.Sequential(
        torch.nn.Linear(5, 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 16),
        torch.nn.ELU(),
        torch.nn.Linear(16, 1),
    )
    critic.requires_grad_(False)
    obs = torch.randn(17, 13, 5, requires_grad=True)
    reward = torch.randn(16, 13, requires_grad=True)
    logits = torch.randn(16, 13, requires_grad=True)
    done = logits.sigmoid()
    trunc = torch.zeros(16, 13, dtype=torch.bool)
    trunc[3, 0], trunc[9, 4] = True, True
    data = {"obs": obs, "reward": reward, "done": done, "truncated": trunc}
    a = evaluate(critic, obs)
    b = evaluate(critic, obs, chunk=37, recompute=True)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(
        returns(data, critic, 0.99, a).sum(), (obs, reward, logits), retain_graph=True
    )
    gb = torch.autograd.grad(
        returns(data, critic, 0.99, b).sum(), (obs, reward, logits)
    )
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y)


def test_real_actor_adam_and_critic_update_equivalence(tmp_path):
    from test_repaired_joint import repaired

    a = repaired(tmp_path)
    a.cfg.horizon = 16
    a.cfg.actor_microbatch = 4
    a.cfg.batch_imagination_critic = True
    b = copy.deepcopy(a)
    b.cfg.critic_value_chunk = 7
    b.cfg.checkpoint_actor_critic = True
    rng = torch.get_rng_state()
    a.update_actor()
    torch.set_rng_state(rng)
    b.update_actor()
    for name in ("actor", "critic"):
        for k, value in getattr(a, name).state_dict().items():
            torch.testing.assert_close(value, getattr(b, name).state_dict()[k])
        for p, q in zip(getattr(a, name).parameters(), getattr(b, name).parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad)
