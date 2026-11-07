"""CPU: compact conditioning never approximates analytical physical metadata."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch
from reflex.world_model.compact_context import (
    CompactContextReplay,
    ContextSpec,
    audit_randomization,
)
from reflex.world_model.models import (
    CompactContextWorldModel,
    DeterministicEventWorldModel,
)
from reflex.world_model.replay import Replay
from reflex.world_model.trainer import TrainConfig, Trainer
from test_wm_compute_acceleration import conditional
from test_world_model import builder, robot_state
from torch import nn


def layout():
    fk = builder().fk
    names = ("pelvis",) + tuple(child for _, child, jid in fk.nodes if jid >= 0)
    schema = (
        ("box.dimensions", 3),
        ("box.masses", 1),
        ("box.inertias", 9),
        ("box.coms", 7),
        ("box.material_properties", 3),
        ("robot.masses", 30),
        ("robot.inertias", 270),
        ("robot.coms", 210),
        ("robot.material_properties", 87),
        *(
            ("robot." + p, 29)
            for p in (
                "joint_stiffness",
                "joint_damping",
                "joint_armature",
                "joint_friction_coeff",
                "joint_effort_limits",
            )
        ),
    )
    return ContextSpec(schema, names)


def context(spec, n=6):
    g = torch.Generator().manual_seed(791)
    x = torch.randn(n, spec.full_dim, generator=g) * 0.02
    x[:, :3] = torch.tensor([0.31, 0.26, 0.21])
    x[:, 3] = 0.18
    for name in ("box.coms", "robot.coms"):
        lo, hi = spec.slices[name]
        v = x[:, lo:hi].view(n, -1, 7)
        v[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    x[:, spec.slices["robot.masses"][0] : spec.slices["robot.masses"][1]] = (
        1 + torch.rand(n, 30, generator=g)
    )
    return x


def transition(spec, j=0, n=6):
    x = context(spec, n)
    x[:, 20:23] += j * 0.01  # reset material changes only selected neural fields
    return {
        "context": x,
        "state": robot_state(n),
        "valid": torch.ones(n, dtype=torch.bool),
        "start_valid": torch.ones(n, dtype=torch.bool),
        "boundary": torch.zeros(n, dtype=torch.bool),
        "terminated": torch.zeros(n, dtype=torch.bool),
        "timeout": torch.zeros(n, dtype=torch.bool),
        "phase": torch.full((n,), j),
        "reference_length": torch.full((n,), 100),
    }


def test_actual_schema_projection_dimensions_and_material_shapes():
    s = layout()
    assert s.full_dim == 765 and s.wm_dim == 127
    assert [len(v) for _, v in s.groups] == [3, 1, 3, 30, 3, 87]
    model = CompactContextWorldModel(s, hidden_layers=[16] * 3, com_indices=(323, 13))
    assert model.net[0].in_features == 86 + 29 + 127 + 4 == 246
    assert model.inputs.mean.numel() == 246
    assert set(range(4, 20)).isdisjoint(
        s.columns
    )  # inertia and box COM are sidecar only
    with pytest.raises(ValueError):
        ContextSpec((*s.schema, ("random_gain", 29)), s.body_names)


@pytest.mark.parametrize("steps", [3, 8, 19])
def test_ring_windows_rng_and_full_physical_context_exact(steps):
    s = layout()
    raw = Replay(8)
    small = CompactContextReplay(8, s)
    for j in range(steps):
        tr = transition(s, j)
        if j == 2:
            tr["context"][:, 4] += (
                1.0  # actual omitted inertia change -> new immutable version
            )
        if j == 4:
            tr["context"][0, 5] = -0.0
        if j == 5:
            tr["boundary"][2] = True
        raw.add(tr)
        small.add(tr)
    assert "context" not in small.data
    assert small.data["wm_context"].shape[-1] == 127
    assert torch.equal(
        raw.ordered("context").view(torch.int32),
        small.ordered("context").view(torch.int32),
    )
    for sample in (
        lambda r: r.sample(8, 2),
        lambda r: r.sample_variable_windows(8, 8),
        lambda r: r.sample_starts(8),
    ):
        torch.manual_seed(721)
        a = sample(raw)
        rng = torch.get_rng_state()
        torch.manual_seed(721)
        b = sample(small)
        assert torch.equal(rng, torch.get_rng_state())
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    assert small.used <= 24


def test_bank_gc_resume_and_strict_formats():
    s = layout()
    r = CompactContextReplay(2, s)
    for j in range(30):
        tr = transition(s, j)
        tr["context"][:, 4] += j  # force actual immutable-version churn
        r.add(tr)
    r._compact_bank()
    before = r.ordered("context").clone()
    state = r.state_dict()
    restored = CompactContextReplay(2, s)
    restored.load_state_dict(state)
    torch.testing.assert_close(before, restored.ordered("context"), rtol=0, atol=0)
    tr = transition(s, 31)
    restored.add(tr)
    torch.testing.assert_close(
        restored.ordered("context")[-1], tr["context"], rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="CompactContextReplay"):
        Replay(2).load_state_dict(state)
    with pytest.raises(ValueError, match="schema mismatch"):
        restored.load_state_dict(Replay(2).state_dict())
    assert restored.memory_report()["physical_bank_versions"] > 0


def test_integrator_uses_full_com_not_compact_column_offsets():
    s = layout()
    ctx = context(s, 4)
    model = CompactContextWorldModel(s, hidden_layers=[16] * 3, com_indices=(323, 13))
    old = DeterministicEventWorldModel(
        765, hidden_layers=[16] * 3, com_indices=(323, 13)
    )

    class FixedDelta(nn.Module):
        def forward(self, x):
            return x.new_full((*x.shape[:-1], 82), 0.03)

    model.net = FixedDelta()
    old.net = FixedDelta()
    state = robot_state(4).requires_grad_()
    a = torch.zeros(4, 29)
    e = torch.zeros(4, 4)
    expected = old.predict_physical(state, a, ctx, e)
    actual = model.predict_physical(state, a, ctx, e)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ga = torch.autograd.grad(actual.sum(), state, retain_graph=True)[0]
    gb = torch.autograd.grad(expected.sum(), state)[0]
    torch.testing.assert_close(ga, gb, rtol=0, atol=0)


def compact_trainer(tmp_path, mode="stage1_dr_v1"):
    source, b = conditional(tmp_path)
    s = layout()
    task = copy.deepcopy(source.task)
    task.body_names = list(s.body_names)
    task.context_slices = {"robot.coms": s.slices["robot.coms"]}
    cfg = copy.deepcopy(source.cfg)
    cfg.wm_context_mode = mode
    contract = {
        **source.contract,
        "context_schema": list(s.schema),
        "body_names": list(s.body_names),
        "context_dim": 765,
        "com_indices": (323, 13),
    }
    scheduler = copy.deepcopy(source.event_scheduler)
    scheduler.task = task
    t = Trainer(
        copy.deepcopy(source.bridge),
        copy.deepcopy(source.action_map),
        source.prior,
        765,
        6,
        source.reward_names,
        contract,
        cfg=cfg,
        task=task,
        event_scheduler=scheduler,
    )
    b["context"] = context(s, 4)
    return t, b


def test_analytical_rewards_termination_and_events_unchanged(tmp_path):
    t, b = compact_trainer(tmp_path)
    r = t.replay
    tr = transition(r.spec, 0, 4)
    r.add(tr)
    exact = r.ordered("context")[0]
    a = {**b, "context": tr["context"]}
    b = {**b, "context": exact}
    before = robot_state(4)
    after = before.clone()
    after[:, 39] = 0.2

    def evaluate(row):
        return t.task.evaluate_step(
            row,
            before,
            after,
            row["action"],
            row["previous_action"],
            row["phase"],
            row["task_history"],
            row["episode_steps"],
            row["next_external_event_seconds"],
        )

    x, y = evaluate(a), evaluate(b)
    for key in ("rewards", "terminated", "timeout", "termination_features", "history"):
        torch.testing.assert_close(getattr(x, key), getattr(y, key), rtol=0, atol=0)
    rng = t.event_scheduler.rng.clone()
    pool = before.new_empty(4, 0, 4)

    def event(row):
        t.event_scheduler.rng = rng.clone()
        return t.event_scheduler.step(
            row,
            after,
            t.bridge.fk(after),
            row["event_state"],
            pool,
            row["episode_steps"] + 1,
            row["episode_steps"] + 1,
        )

    p, e, _ = event(a)
    q, f, _ = event(b)
    torch.testing.assert_close(p.targets, q.targets, rtol=0, atol=0)
    assert torch.equal(p.mask, q.mask)
    torch.testing.assert_close(e, f, rtol=0, atol=0)


def test_dr_audit_rejects_unrepresented_randomization():
    def randomize_joint_gains():
        pass

    term = NS(func=randomize_joint_gains, params={"asset_cfg": NS(name="robot")})
    manager = NS(active_terms={"startup": ["gains"]}, get_term_cfg=lambda _: term)
    with pytest.raises(ValueError, match="Uncovered"):
        audit_randomization(NS(event_manager=manager))


def fill(t, b, steps=40):
    spec = layout()
    state = robot_state(4)
    event = b["event_state"]
    pool = state.new_empty(4, 0, 4)
    for j in range(steps):
        bb = {
            **b,
            "state": state,
            "context": context(spec, 4),
            "phase": torch.full((4,), j),
            "episode_steps": torch.full((4,), j),
            "common_step": torch.full((4,), j),
            "event_state": event,
            "launched": t.event_scheduler.codec.mode(event),
            "learn_external_events": True,
            "event_conditioned": True,
        }
        bb["context"][:, 20:23] += j * 0.001
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
        plan, ce, pool = t.event_scheduler.step(
            bb,
            physical,
            t.bridge.fk(physical),
            event,
            pool,
            bb["episode_steps"] + 1,
            bb["common_step"] + 1,
        )
        post = t.model.predict_post(physical, plan.targets, plan.mask)
        tr = {k: v for k, v in bb.items() if torch.is_tensor(v)}
        tr.update(
            next_state=post,
            physical_next_state=physical,
            event_targets=plan.targets,
            event_mask=plan.mask,
            event_pool=pool[0],
            after_launched=t.event_scheduler.codec.mode(ce),
            next_event_state=ce,
            external_boundary=plan.mask.any(-1),
            valid=torch.ones(4, dtype=torch.bool),
            boundary=result.terminated | result.timeout,
            terminated=result.terminated,
            timeout=result.timeout,
            reward_terms=result.rewards,
            termination_features=result.termination_features,
            contact_known=torch.ones(4, dtype=torch.bool),
            contact_phase=torch.full((4,), 1 if 10 <= j <= 20 else 2 if j > 20 else 0),
        )
        t.replay.add(tr)
        state, event = post, ce


def test_compact_fresh_training_checkpoint_and_no_weight_migration(tmp_path):
    t, b = compact_trainer(tmp_path)
    fill(t, b)
    assert not t.model_opt.state
    before_actor = copy.deepcopy(t.actor.state_dict())
    result = t.fit_model()
    assert result["termination/bce_updates_wm"] == 1
    assert t.model.net[0].in_features == 246
    assert "context" not in t.replay.data
    p = tmp_path / "compact.pt"
    t.save(p)
    context_before = t.replay.ordered("context").clone()
    t.load(p)
    torch.testing.assert_close(
        context_before, t.replay.ordered("context"), rtol=0, atol=0
    )
    t.fit_model()
    assert all(torch.equal(v, before_actor[k]) for k, v in t.actor.state_dict().items())
    with pytest.raises(ValueError, match="fresh initialization"):
        t.warm_start_macro(tmp_path / "missing_old.pt")
    full, _ = compact_trainer(tmp_path, mode="full")
    with pytest.raises(ValueError):
        full.load(p)


def test_fixed_context_guard_reports_hidden_controller_changes():
    from reflex.world_model.compact_context import FixedContextGuard

    s = layout()
    x = context(s)
    guard = FixedContextGuard(s)
    guard.check(x)
    y = x.clone()
    y[:, 20:23] += 0.1
    guard.check(y)
    y = y.clone()
    y[:, 4] += 0.1
    guard.check(y)  # derived inertia is kept, not assumed fixed
    y = y.clone()
    y[:, s.slices["robot.joint_stiffness"][0]] += 1
    with pytest.raises(ValueError, match="fixed physical"):
        guard.check(y)


def test_fresh_full_compact_ablation_smoke_and_policy_invariance(tmp_path):
    from reflex.world_model.context_ablation import run

    source, b = compact_trainer(tmp_path, mode="full")
    fill(source, b)
    original = copy.deepcopy(source.model.state_dict())
    out = tmp_path / "ablation"
    out.mkdir()
    manager = NS(active_terms={})
    report = run(
        source,
        NS(env=NS(event_manager=manager)),
        out,
        rounds=1,
        collect_steps=0,
        validate_every=1,
        batch=4,
    )
    assert report["completed"] and report["actor_updates"] == 0
    assert all(v["trainer_released"] for v in report["results"].values())
    exported = torch.load(out / "compact/checkpoint_final.pt", weights_only=False)
    assert exported["contract"]["wm_context_audit"]["context"]["wm_dim"] == 127
    cfg = TrainConfig(**exported["config"])
    scheduler = copy.deepcopy(source.event_scheduler)
    task = copy.deepcopy(source.task)
    scheduler.task = task
    restored = Trainer(
        source.bridge,
        source.action_map,
        source.prior,
        765,
        6,
        source.reward_names,
        exported["contract"],
        cfg=cfg,
        task=task,
        event_scheduler=scheduler,
    )
    restored.load(out / "compact/checkpoint_final.pt", reinitialize_replay=True)
    assert restored.model.net[0].in_features == 246
    assert report["results"]["full765"]["wm_input_dim"] == 884
    assert report["results"]["compact"]["wm_input_dim"] == 246
    assert (
        report["results"]["compact"]["records"][-1][
            "validation_v24/8/contact_labels_available"
        ]
        == 1.0
    )
    assert (
        report["results"]["compact"]["records"][-1][
            "validation_v24/conditioned/8/contact_samples"
        ]
        > 0
    )
    assert all(
        torch.equal(v, original[k]) for k, v in source.model.state_dict().items()
    )
    assert (out / "full765/checkpoint_final.pt").exists()
    assert (out / "compact/checkpoint_final.pt").exists()


@pytest.mark.parametrize(
    "function",
    [
        "randomize_motion_yaw",
        "randomize_init_state",
        "randomize_cmd_time_steps_mixture",
    ],
)
def test_dr_audit_distinguishes_native_reset_state_from_physical_parameters(function):
    fn = NS(__name__=function, __module__="reflex.envs.base.catch_terms_v2")
    term = NS(func=fn, params={})
    manager = NS(active_terms={"reset": [function]}, get_term_cfg=lambda _: term)
    result = audit_randomization(NS(event_manager=manager))
    assert result["events"][0]["conditioning"] == "reset_state_or_reference"
    manager.active_terms = {"interval": [function]}
    with pytest.raises(ValueError, match="must remain at reset"):
        audit_randomization(NS(event_manager=manager))


def test_dr_audit_inherited_motion_yaw_source():
    fn = NS(
        __name__="randomize_motion_yaw",
        __module__="whole_body_tracking.tasks.tracking.mdp.catch_extras",
    )
    term = NS(func=fn, params={})
    manager = NS(
        active_terms={"reset": ["randomize_motion_yaw"]}, get_term_cfg=lambda _: term
    )
    result = audit_randomization(NS(event_manager=manager))
    assert result["events"][0]["conditioning"] == "reset_state_or_reference"
    fn.__module__ = "unknown.custom_module"
    with pytest.raises(ValueError, match="Uncovered"):
        audit_randomization(NS(event_manager=manager))
