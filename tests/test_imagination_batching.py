"""Experimental batching must preserve gradients and optimizer updates."""

import copy

import torch
from reflex.world_model.joint import critic_targets, returns
from test_repaired_joint import repaired


def test_batched_amp_trajectory_and_actor_adam(tmp_path):
    old = repaired(tmp_path)
    old.cfg.horizon = 32
    old.cfg.actor_steps = 1
    new = copy.deepcopy(old)
    new.cfg.batch_imagination_amp = True
    new.cfg.batch_imagination_critic = True
    start = old.sample(starts=True, n=4)
    outputs = []
    grads = []
    for trainer in (old, new):
        imagined = trainer.imagine({k: v.clone() for k, v in start.items()})
        values = None
        if trainer.cfg.batch_imagination_critic:
            obs = imagined["critic_obs"]
            values = trainer.critic(obs.flatten(0, 1)).reshape(obs.shape[:2])
        target = critic_targets(trainer, imagined, values)
        loss = -returns(imagined, trainer.critic, trainer.cfg.gamma, values).mean()
        trainer.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in trainer.actor.parameters()]))
        trainer.actor_opt.step()
        outputs.append((imagined, target))
    for key in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key])
    torch.testing.assert_close(outputs[0][1], outputs[1][1])
    torch.testing.assert_close(grads[0], grads[1])
    for p, q in zip(old.actor.parameters(), new.actor.parameters()):
        torch.testing.assert_close(p, q)


def test_batched_critic_truncation_and_survival_gradient():
    torch.manual_seed(17)
    critic = torch.nn.Sequential(
        torch.nn.Linear(5, 12), torch.nn.ELU(), torch.nn.Linear(12, 1)
    )
    obs = torch.randn(33, 4, 5, requires_grad=True)
    reward = torch.randn(32, 4, requires_grad=True)
    done = torch.sigmoid(torch.randn(32, 4, requires_grad=True))
    truncated = torch.zeros(32, 4, dtype=torch.bool)
    truncated[2, 0] = True
    truncated[15, 1] = True
    data = {"obs": obs, "reward": reward, "done": done, "truncated": truncated}
    a = returns(data, critic, 0.99)
    values = critic(obs.flatten(0, 1)).reshape(33, 4)
    b = returns(data, critic, 0.99, values)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.sum(), (obs, reward, done), retain_graph=True)
    gb = torch.autograd.grad(b.sum(), (obs, reward, done))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y)


def test_paired_probe_restores_weights_flags_and_rng(tmp_path):
    from reflex.world_model.imagination_batching import check
    from test_compact_context import compact_trainer, fill

    trainer, batch = compact_trainer(tmp_path, mode="full")
    fill(trainer, batch)
    trainer.cfg.horizon = 4
    trainer.cfg.actor_steps = 1
    trainer.cfg.gradient_audit_every = 0
    before = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = check(trainer, tmp_path, repeats=1)
    assert report["completed"] and report["numeric_passed"]
    assert trainer.actor_updates == 0
    assert not trainer.cfg.batch_imagination_amp
    assert not trainer.cfg.batch_imagination_critic
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for k, value in trainer.actor.state_dict().items():
        torch.testing.assert_close(value, before[k], rtol=0, atol=0)
