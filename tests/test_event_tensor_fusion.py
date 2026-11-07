"""Facade clones/codec fusion must preserve mutation isolation and gradients."""

import copy

import torch
from reflex.world_model.events_v24.execution_v30 import configure
from reflex.world_model.events_v24.tensors_v30 import configure as configure_tensors
from reflex.world_model.joint import returns
from test_analytic_soft_termination import new_trainer
from test_compact_context import compact_trainer, fill
from test_v24_events import setup


def test_compiled_facade_state_plan_rng_gradients_and_input_isolation(tmp_path):
    trainer, row, scheduler, event = setup(tmp_path)
    outputs = []
    for compiled in (False, True):
        runtime = copy.deepcopy(scheduler)
        configure(
            runtime, track_pose=True, compact_pools=True, collapse_full_subsets=True
        )
        configure_tensors(runtime, compiled, "aot_eager")
        physical = row["state"].clone().requires_grad_()
        state = event.clone().requires_grad_()
        before = state.detach().clone()
        pools = physical.new_empty(4, 0, 4)
        torch.manual_seed(3017)
        plan, updated, _ = runtime.step(
            row,
            physical,
            trainer.bridge.fk(physical),
            state,
            pools,
            torch.ones(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
            isolated_rng=False,
        )
        loss = plan.targets.square().sum() + updated.square().sum() * 0.01
        gradient = torch.autograd.grad(loss, (physical, state))
        torch.testing.assert_close(state.detach(), before, atol=0, rtol=0)
        outputs.append(
            (plan.targets, plan.mask, updated, gradient, torch.get_rng_state())
        )
    for a, b in zip(outputs[0][:3], outputs[1][:3]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(outputs[0][3], outputs[1][3]):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(outputs[0][4], outputs[1][4], atol=0, rtol=0)
    torch._dynamo.reset()


def test_compiled_event_facade_h32_and_actor_adam(tmp_path):
    source, row = compact_trainer(tmp_path, mode="full")
    trainers = []
    for enabled in (False, True):
        torch.manual_seed(3098)
        trainer = new_trainer(
            source,
            minimal_event_view=True,
            compact_event_inputs=True,
            reuse_post_event_fk=True,
            compact_event_pool_updates=True,
            collapse_full_event_subsets=True,
            batch_imagination_amp=True,
            batch_imagination_amp_features=True,
            batch_imagination_critic=True,
            compile_event_tensors=enabled,
            compile_backend="aot_eager",
        )
        fill(trainer, row)
        trainer.cfg.horizon = 32
        trainers.append(trainer)
    torch.manual_seed(3099)
    start = trainers[0].sample(starts=True, n=4)
    start["episode_steps"][:2] = trainers[0].task.max_steps - 4
    results = []
    for trainer in trainers:
        result = trainer.imagine({k: v.clone() for k, v in start.items()}, trace=True)
        objective = returns(result, trainer.critic, trainer.cfg.gamma).mean()
        trainer.actor_opt.zero_grad(set_to_none=True)
        objective.backward()
        gradients = [p.grad.clone() for p in trainer.actor.parameters()]
        trainer.actor_opt.step()
        results.append((result, gradients))
    for key in results[0][0]:
        torch.testing.assert_close(
            results[0][0][key], results[1][0][key], atol=0, rtol=0
        )
    for a, b in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(a, b)
    for a, b in zip(trainers[0].actor.parameters(), trainers[1].actor.parameters()):
        torch.testing.assert_close(a, b)
    torch._dynamo.reset()


def test_tensor_probe_restores_codec_and_actual_optimizer_updates(tmp_path):
    from reflex.world_model.imagination_batching import check

    source, row = compact_trainer(tmp_path, mode="full")
    trainer = new_trainer(
        source,
        minimal_event_view=True,
        compact_event_inputs=True,
        reuse_post_event_fk=True,
        compact_event_pool_updates=True,
        collapse_full_event_subsets=True,
        batch_imagination_amp=True,
        batch_imagination_amp_features=True,
        batch_imagination_critic=True,
        compile_backend="aot_eager",
    )
    fill(trainer, row)
    trainer.cfg.horizon = 4
    trainer.cfg.actor_steps = 1
    trainer.cfg.gradient_audit_every = 0
    report = check(
        trainer,
        tmp_path,
        repeats=1,
        execution_variants=True,
        event_tensor_variants=True,
    )
    assert report["completed"] and report["numeric_passed"], report.get(
        "failure_details"
    )
    assert set(report["variants"]) == {"reference", "event_tensors"}
    assert not trainer.cfg.compile_event_tensors
    assert "decode" not in trainer.event_scheduler.codec.__dict__
    assert not hasattr(trainer.event_scheduler, "tensor_preparer")
    torch._dynamo.reset()
