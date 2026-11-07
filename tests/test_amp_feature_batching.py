"""Pure AMP features may move after trajectory generation, preserving BPTT."""

import copy

import pytest
import torch
from reflex.world_model.joint import returns
from test_analytic_soft_termination import new_trainer
from test_compact_context import compact_trainer, fill


@pytest.mark.parametrize("truncate", [False, True])
def test_h32_amp_features_trajectory_gradient_and_complete_updates(tmp_path, truncate):
    source, row = compact_trainer(tmp_path, mode="full")
    baseline = new_trainer(
        source,
        batch_imagination_amp=True,
        batch_imagination_critic=True,
        minimal_event_view=True,
        compact_event_inputs=True,
        reuse_post_event_fk=True,
        compact_event_pool_updates=True,
    )
    fill(baseline, row)
    baseline.cfg.horizon = 32
    baseline.cfg.actor_steps = 1
    baseline.cfg.gradient_audit_every = 0
    candidate = copy.deepcopy(baseline)
    candidate.cfg.batch_imagination_amp_features = True
    start = baseline.sample(starts=True, n=4)
    if truncate:
        start["episode_steps"][:2] = baseline.task.max_steps - 2
    outputs, gradients = [], []
    for trainer in (baseline, candidate):
        initial_rng = trainer.event_scheduler.rng.clone()
        imagined = trainer.imagine({k: v.clone() for k, v in start.items()}, trace=True)
        loss = returns(imagined, trainer.critic, trainer.cfg.gamma).mean()
        gradients.append(torch.autograd.grad(loss, tuple(trainer.actor.parameters())))
        outputs.append(imagined)
        trainer.event_scheduler.rng = initial_rng
        trainer.sample = lambda *args, **kwargs: {
            k: v.clone() for k, v in start.items()
        }
        torch.manual_seed(3082)
        trainer.update_actor()
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b)
    for name in ("actor", "critic"):
        for key, value in getattr(baseline, name).state_dict().items():
            torch.testing.assert_close(
                getattr(candidate, name).state_dict()[key], value
            )
