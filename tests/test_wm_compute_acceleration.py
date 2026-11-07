"""Pure CPU equivalence for the high-cost WM computation candidates."""

import copy

import pytest
import torch
from reflex.world_model.models import PrecisionMLP
from reflex.world_model.physics import state_error
from reflex.world_model.trainer import Trainer
from test_repaired_joint import TRACKED, repaired
from test_v24_events import setup
from test_world_model import builder, robot_state
from torch import nn


def test_partial_fk_values_and_gradients_are_original_recursion():
    torch.set_num_threads(1)
    fk = builder().fk
    s = robot_state(4)
    s[:, 7:36] = torch.randn(4, 29) * 0.2
    s.requires_grad_()
    for names in (("torso_link",), tuple(TRACKED)):
        original = fk(s)
        actual = fk.subset(s, names)
        for a, b in zip(original, actual):
            for n in names:
                torch.testing.assert_close(a[n], b[n], atol=0, rtol=0)

        def objective(result, names=names):
            return sum(
                result[0][n].square().sum() + result[1][n].square().sum() for n in names
            )

        ga = torch.autograd.grad(objective(original), s, retain_graph=True)[0]
        gb = torch.autograd.grad(objective(actual), s)[0]
        torch.testing.assert_close(ga, gb, atol=0, rtol=0)


def test_minimal_fk_full_training_and_resume_equal(tmp_path):
    t = repaired(tmp_path)
    # Add predicted-state termination training, matching the joint loss path.
    t.cfg.joint_predicted_termination = True
    t.cfg.minimal_termination_fk = False
    p = tmp_path / "initial.pt"
    t.save(p)
    torch.manual_seed(87)
    t.fit_model()
    expected = {
        n: copy.deepcopy(getattr(t, n).state_dict())
        for n in ("model", "termination", "model_opt", "termination_opt")
    }
    target_cfg = copy.deepcopy(t.cfg)
    target_cfg.minimal_termination_fk = True
    t.cfg = target_cfg
    t.load(p)
    torch.manual_seed(87)
    t.fit_model()

    def compare(a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                compare(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b

    for n, value in expected.items():
        compare(value, getattr(t, n).state_dict())


def conditional(tmp_path):
    old, b, runtime, event = setup(tmp_path)
    cfg = copy.deepcopy(old.cfg)
    cfg.learn_external_events = cfg.condition_external_events = (
        cfg.deterministic_event_apply
    ) = True
    cfg.joint_predicted_termination = cfg.minimal_termination_fk = True
    cfg.state_loss_reduction = "sum"
    t = Trainer(
        old.bridge,
        old.action_map,
        old.prior,
        98,
        6,
        old.reward_names,
        old.contract,
        cfg=cfg,
        task=old.task,
        event_scheduler=runtime,
    )
    t.model.warm_start_physical(old.model.state_dict())
    # Avoid the all-zero residual head making BF16/gradient tests trivial.
    with torch.no_grad():
        t.model.net[-1].weight.normal_(0, 0.001)
    row = {
        **b,
        "event_state": event,
        "episode_steps": torch.zeros(4, dtype=torch.long),
        "target": torch.randn(4, 29) * 0.1,
        "physical_next_state": b["state"].clone(),
        "next_state": b["state"].clone(),
        "event_targets": torch.zeros(4, 84),
        "event_mask": torch.zeros(4, 84, dtype=torch.bool),
        "terminated": torch.tensor([False, True, False, False]),
    }
    # A normal external write remains differentiable and inside the rollout.
    row["event_mask"][:, 78] = True
    row["event_targets"][:, 78] = 3.0
    return t, row


@pytest.mark.parametrize("initial", [False, True])
def test_fused_wm_step_loss_and_joint_gradients(tmp_path, initial):
    t, row = conditional(tmp_path)
    s = row["state"].clone().requires_grad_()
    pre = t.event_scheduler.codec.pre_features(
        row["event_state"], row["episode_steps"], t.task.dt
    )
    p = t.model.predict_physical(s, row["target"], row["context"], pre)
    post = t.model.predict_post(p, row["event_targets"], row["event_mask"])
    features = t.task.predicted_termination_features(
        row,
        s,
        p,
        row["phase"],
        initial=initial,
        fk_before=t.bridge.fk(s),
        fk_after=t.bridge.fk(p),
    )
    pl = state_error(p, row["physical_next_state"]).square().sum(-1)
    dl = nn.functional.binary_cross_entropy_with_logits(
        t.termination(features), row["terminated"].float(), reduction="none"
    )
    expected = (post, pl, dl, state_error(post, row["next_state"]).square().sum(-1))
    params = (s, *t.model.parameters(), *t.termination.parameters())
    eg = torch.autograd.grad((pl + t.cfg.term_loss_weight * dl).mean(), params)
    compiled = torch.compile(
        t.v24_model_training_step, backend="aot_eager", fullgraph=True
    )
    actual = compiled(s, row, initial)
    ag = torch.autograd.grad(
        (actual[1] + t.cfg.term_loss_weight * actual[2]).mean(), params
    )
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    for a, b in zip(eg, ag):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    assert torch.stack([g.square().sum() for g in ag[1:]]).sum() > 0
    # Test the actual clipped Adam update, not just matching loss scalars.
    objects = ("model", "termination", "model_opt", "termination_opt")
    initial_states = {n: copy.deepcopy(getattr(t, n).state_dict()) for n in objects}
    expected_states = None
    for gradients in (eg, ag):
        for n in objects:
            getattr(t, n).load_state_dict(copy.deepcopy(initial_states[n]))
        for parameter, gradient in zip(params[1:], gradients[1:]):
            parameter.grad = gradient.detach().clone()
        nn.utils.clip_grad_norm_(
            params[1:], t.cfg.model_grad_clip, error_if_nonfinite=True
        )
        t.model_opt.step()
        t.termination_opt.step()
        current = {n: copy.deepcopy(getattr(t, n).state_dict()) for n in objects}
        if expected_states is None:
            expected_states = current
        else:

            def compare(a, b):
                if torch.is_tensor(a):
                    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                elif isinstance(a, dict):
                    assert a.keys() == b.keys()
                    for key in a:
                        compare(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    assert len(a) == len(b)
                    for x, y in zip(a, b):
                        compare(x, y)
                else:
                    assert a == b

            compare(current, expected_states)


def test_bf16_scope_fp32_state_parameters_and_optimizer(tmp_path):
    t, row = conditional(tmp_path)
    t.model.net.precision = "bf16"
    t.cfg.wm_mlp_precision = "bf16"
    ns, pl, dl, post = t.v24_model_training_step(row["state"], row, True)
    assert ns.dtype == pl.dtype == dl.dtype == post.dtype == torch.float32
    (pl + t.cfg.term_loss_weight * dl).mean().backward()
    t.model_opt.step()
    t.termination_opt.step()
    assert type(t.actor) is type(t.critic) is type(t.termination.net) is nn.Sequential
    for p in t.model.parameters():
        assert p.dtype == torch.float32 and p.grad.dtype == torch.float32
        assert torch.isfinite(p.grad).all()
    for state in t.model_opt.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    assert isinstance(t.model.net, PrecisionMLP)
    saved = copy.deepcopy(t.model.state_dict())
    path = tmp_path / "bf16_mlp_fp32_checkpoint.pt"
    t.save(path)
    # Precision is an explicit execution choice, never a dtype-changing load.
    t.cfg.wm_mlp_precision = "fp32"
    t.model.net.precision = "fp32"
    t.load(path)
    assert t.model.net.precision == "fp32"
    for k, value in t.model.state_dict().items():
        torch.testing.assert_close(value, saved[k], atol=0, rtol=0)
        assert value.dtype == torch.float32


def test_fused_h8_train_resume_and_h32_event_contract(tmp_path):
    t, row = conditional(tmp_path)
    t.cfg.model_window, t.cfg.horizon = 8, 32
    t.cfg.compile_wm_training_step = True
    t.cfg.compile_backend = "aot_eager"
    t._wm_training_step = torch.compile(
        t.v24_model_training_step, backend="aot_eager", fullgraph=True
    )
    scheduler = t.event_scheduler
    state, event = row["state"].clone(), row["event_state"].clone()
    pools = state.new_empty(4, 0, 4)
    for j in range(16):
        bb = {
            **row,
            "state": state,
            "phase": torch.full((4,), j),
            "episode_steps": torch.full((4,), j),
            "common_step": torch.full((4,), j),
            "launched": scheduler.codec.mode(event),
            "event_state": event,
            "learn_external_events": True,
            "event_conditioned": True,
        }
        physical = state.clone()
        physical[:, 80] -= 0.1962
        physical[:, 71:74] += physical[:, 78:81] * 0.02
        result = t.task.evaluate_step(
            bb,
            state,
            physical,
            bb["action"],
            bb["previous_action"],
            bb["phase"],
            bb["task_history"],
            bb["episode_steps"],
            bb["next_external_event_seconds"],
        )
        plan, after_event, pools = scheduler.step(
            bb,
            physical,
            t.bridge.fk(physical),
            event,
            pools,
            bb["episode_steps"] + 1,
            bb["common_step"] + 1,
        )
        post = t.model.predict_post(physical, plan.targets, plan.mask)
        transition = {k: v for k, v in bb.items() if torch.is_tensor(v)}
        transition.update(
            next_state=post,
            physical_next_state=physical,
            event_pool=pools[0],
            event_targets=plan.targets,
            event_mask=plan.mask,
            next_event_state=after_event,
            external_boundary=plan.mask.any(-1),
            after_launched=scheduler.codec.mode(after_event),
            valid=torch.ones(4, dtype=torch.bool),
            boundary=result.terminated | result.timeout,
            terminated=result.terminated,
            timeout=result.timeout,
            reward_terms=result.rewards,
            next_memory=torch.zeros(4, 6),
            termination_features=result.termination_features,
        )
        t.replay.add(transition)
        state, event = post, after_event
    # The normal launch is strictly inside at least one legal H8 window.
    assert len(t.replay.indices(8)) > 0
    assert (
        ~t.replay.ordered("launched").bool() & t.replay.ordered("after_launched").bool()
    ).any()
    metrics = t.fit_model()
    assert metrics["termination/bce_updates_wm"] == 1.0
    assert t.model_updates == t.termination_updates == t.cfg.model_steps
    path = tmp_path / "fused_h8.pt"
    t.save(path)
    t.load(path)
    assert t.wm_rounds_completed == 1
    # Waiting snapshots, not just flight snapshots, must survive load and launch.
    starts = {
        k: v[0].clone()
        for k, v in t.replay.data.items()
        if k not in t.replay.global_fields
    }
    starts["event_pool"] = (
        t.replay.data["event_pool"][0].unsqueeze(0).expand(4, -1, -1).clone()
    )
    imagined = t.imagine(starts)
    assert not imagined["predicted_launch_probability"][0].bool().any()
    assert imagined["predicted_launch_probability"][1].bool().all()
    assert len(imagined["reward"]) == 32
    updated = t.update_actor()
    assert updated["actor/gradient_norm"] > 0
    assert t.actor_updates == 1


def test_actor_export_remains_original_sequential(tmp_path):
    t = repaired(tmp_path)
    assert type(t.actor) is nn.Sequential
    scripted = torch.jit.script(t.actor)
    x = torch.randn(3, 179)
    torch.testing.assert_close(scripted(x), t.actor(x), atol=0, rtol=0)
