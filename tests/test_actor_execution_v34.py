"""Experimental write organization: values, gradients and actual updates."""

import copy

import pytest
import torch
from reflex.world_model.actor_execution_v34_check import check
from reflex.world_model.diagnostics_v31.minimal import boundary_probe
from reflex.world_model.events_v24.writes import ContiguousWritePlan, WritePlan
from test_actor_diagnostics_v31 import fixture as trainer_fixture


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


@pytest.mark.parametrize("rows", [None, [2, 0], []])
@pytest.mark.parametrize(
    "slots", [list(range(71, 78)), list(range(78, 84)), [7, 9, 12]]
)
def test_repeated_writes_exact_values_and_gradients(rows, slots):
    state = torch.randn(3, 84)
    ids = None if rows is None else torch.tensor(rows, dtype=torch.long)
    n = 3 if rows is None else len(rows)
    values = [torch.randn(n, len(slots), requires_grad=True) for _ in range(2)]
    masks = [torch.arange(n)[:, None] % 2 == k for k in (0, 1)]
    outputs, grads = [], []
    for cls in (WritePlan, ContiguousWritePlan):
        plan = cls(state)
        for v, mask in zip(values, masks):
            plan.record(None, v, ids, slots, mask)
        output = torch.where(plan.mask, plan.targets, state)
        outputs.append((output, plan.mask))
        grads.append(torch.autograd.grad(output.square().sum(), values))
    for a, b in zip(outputs[0], outputs[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(grads[0], grads[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_select_merge_independent_storage_and_gradients():
    x = torch.randn(4, 84, requires_grad=True)
    outputs = []
    for cls in (WritePlan, ContiguousWritePlan):
        plan = cls(x)
        plan.record(None, x[:, 78:84], None, list(range(78, 84)))
        ids = torch.tensor([3, 1])
        child = plan.select(ids)
        assert child.targets.data_ptr() != plan.targets.data_ptr()
        child.record(None, child.targets[:, 78:84] * 2, None, list(range(78, 84)))
        plan.merge(ids, child)
        outputs.append(plan.targets)
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    grads = [torch.autograd.grad(v.square().sum(), x)[0] for v in outputs]
    torch.testing.assert_close(*grads, atol=0, rtol=0)


def test_real_rollout_adam_and_restoration(tmp_path):
    t = trainer_fixture(tmp_path)
    t.cfg.compact_actor_event_inputs = True
    t.cfg.packed_pose_carry = True
    t.update_actor()
    rng = torch.get_rng_state().clone()
    opt = copy.deepcopy(t.actor_opt.state_dict())
    result = check(t, tmp_path / "check", repeats=2, warmups=1)
    assert result["completed"] and result["numeric_passed"], result
    assert result["performance_usable"]
    assert not hasattr(t.event_scheduler, "event_plan_factory")
    torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
    for key, value in opt["state"].items():
        for name, expected in value.items():
            torch.testing.assert_close(
                t.actor_opt.state_dict()["state"][key][name], expected, atol=0, rtol=0
            )


def test_same_input_probe_identifies_first_drift():
    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wm_call = None

        def initial(self):
            return (torch.zeros(2, 1),) * 3

        def forward(self, state, previous, aux):
            return state + 1, previous + 1, aux + 1

    step = Step()
    assert boundary_probe(step, step, 4)["first_failed_step"] is None

    def changed(state, previous, aux):
        a, b, c = step(state, previous, aux)
        return a + (state >= 2).float(), b, c

    assert boundary_probe(step, changed, 4)["first_failed_step"] == 3


@pytest.mark.parametrize(
    "callback_name", ["toy_control", "robot_velocity_event", "robot_joint_event"]
)
def test_scheduler_h32_outputs_rng_and_robot_dependent_launch_gradient(
    tmp_path, callback_name
):
    from test_actor_execution_v30 import robot_joint_event, robot_velocity_event
    from test_v24_events import setup, toy_control

    callbacks = {
        "toy_control": toy_control,
        "robot_velocity_event": robot_velocity_event,
        "robot_joint_event": robot_joint_event,
    }
    t, b, scheduler, initial = setup(tmp_path)
    scheduler.terms[0].function = callbacks[callback_name]
    outputs = []
    for cls in (WritePlan, ContiguousWritePlan):
        scheduler.event_plan_factory = cls
        scheduler.rng = torch.Generator().manual_seed(24).get_state()
        state = initial.clone()
        physical = b["state"].clone().requires_grad_()
        pools = torch.empty(len(physical), 0, len(physical))
        records = []
        launch_gradient = None
        for index in range(32):
            episode = torch.full((len(physical),), index + 1)
            plan, state, pools = scheduler.step(
                b, physical, t.bridge.fk(physical), state, pools, episode, episode
            )
            if index == 1:
                launch_gradient = torch.autograd.grad(
                    plan.targets[:, 78:81].sum(), physical, retain_graph=True
                )[0]
            records.extend(
                [plan.targets.detach().clone(), plan.mask.clone(), state.clone()]
            )
            physical = torch.where(plan.mask, plan.targets, physical)
        assert scheduler.codec.mode(state).all()
        assert launch_gradient[:, :3].eq(1).all()
        outputs.append((records, scheduler.rng.clone(), launch_gradient))
    for a, b in zip(outputs[0][0], outputs[1][0]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(outputs[0][1:], outputs[1][1:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_contiguous_recorder_fullgraph_forward_and_backward():
    def write(state, values, active):
        plan = ContiguousWritePlan(state)
        plan.record(None, values, None, list(range(78, 84)), active)
        return torch.where(plan.mask, plan.targets, state)

    state = torch.randn(3, 84, requires_grad=True)
    target = torch.randn(3, 6, requires_grad=True)
    active = torch.tensor([[True], [False], [True]])
    expected = write(state, target, active)
    compiled = torch.compile(write, fullgraph=True, backend="aot_eager", dynamic=False)
    actual = compiled(state, target, active)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    a = torch.autograd.grad(actual.square().sum(), (state, target))
    b = torch.autograd.grad(expected.square().sum(), (state, target))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


@pytest.mark.parametrize("existing_factory", [None, ContiguousWritePlan])
def test_checker_failure_restores_factory_flags_parameters_and_rng(
    tmp_path, monkeypatch, existing_factory
):
    from reflex.world_model.diagnostics_v31.state import digest

    t = trainer_fixture(tmp_path)
    if existing_factory is not None:
        t.event_scheduler.event_plan_factory = existing_factory
    actor = digest(t.actor.state_dict())
    rng = torch.get_rng_state().clone()
    original_flag = t.cfg.contiguous_event_writes

    def fail():
        torch.rand(3)
        with torch.no_grad():
            next(t.actor.parameters()).add_(1)
        raise RuntimeError("injected update failure")

    monkeypatch.setattr(t, "update_actor", fail)
    with pytest.raises(RuntimeError, match="injected"):
        check(t, tmp_path / "failed", repeats=1, warmups=1)
    assert digest(t.actor.state_dict()) == actor
    assert getattr(t.event_scheduler, "event_plan_factory", None) is existing_factory
    assert t.cfg.contiguous_event_writes == original_flag
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)


def test_legacy_checkpoint_and_enabled_factory_preserve_weights_and_adam(tmp_path):
    from reflex.world_model.diagnostics_v31.state import digest
    from test_actor_diagnostics_v31 import assert_tree_equal
    from test_analytic_soft_termination import new_trainer

    source = trainer_fixture(tmp_path)
    source.update_actor()
    path = tmp_path / "old.pt"
    source.save(path)
    saved = torch.load(path, weights_only=False)
    saved["config"].pop("contiguous_event_writes")
    torch.save(saved, path)
    t = new_trainer(source, contiguous_event_writes=True)
    assert t.event_scheduler.event_plan_factory is ContiguousWritePlan
    t.load(path)
    assert digest(t.actor.state_dict()) == digest(source.actor.state_dict())
    assert_tree_equal(t.actor_opt.state_dict(), source.actor_opt.state_dict())


def test_source_pinned_migration_is_atomic_and_rejects_unknown_math():
    import hashlib
    from pathlib import Path

    from reflex.world_model.event_execution_migration import (
        BASE_RUNTIME,
        BASE_WRITES,
        V34_RUNTIME,
        V34_WRITES,
        migrate_sources,
    )
    from reflex.world_model.events_v24 import runtime, writes

    assert (
        hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest() == V34_RUNTIME
    )
    assert hashlib.sha256(Path(writes.__file__).read_bytes()).hexdigest() == V34_WRITES
    old = {
        "runtime.py": BASE_RUNTIME,
        "writes.py": BASE_WRITES,
        "event_math.py": "same",
    }
    new = {"runtime.py": V34_RUNTIME, "writes.py": V34_WRITES, "event_math.py": "same"}
    untouched = copy.deepcopy(old)
    with pytest.raises(ValueError, match="Unverified"):
        migrate_sources(old, {**new, "event_math.py": "unexpected"}, "ckpt")
    assert old == untouched
    report = migrate_sources(old, new, "ckpt")
    assert old == new and report["optimizers_preserved"]


def test_previous_source_checkpoint_requires_explicit_migration(tmp_path):
    from reflex.world_model.diagnostics_v31.state import digest
    from reflex.world_model.event_execution_migration import BASE_RUNTIME, BASE_WRITES

    t = trainer_fixture(tmp_path)
    t.update_actor()
    path = tmp_path / "previous.pt"
    t.save(path)
    saved = torch.load(path, weights_only=False)
    sources = saved["learning_contract"]["event_scheduler"]["source_sha256"]
    sources["runtime.py"], sources["writes.py"] = BASE_RUNTIME, BASE_WRITES
    saved["config"].pop("contiguous_event_writes")
    torch.save(saved, path)
    before = digest(t.actor.state_dict())
    with pytest.raises(ValueError, match="learning layout"):
        t.load(path)
    t.load(path, migrate_event_execution=True)
    assert digest(t.actor.state_dict()) == before
    assert t.initialization["event_execution_migration"]["optimizers_preserved"]


def test_same_input_probe_preserves_recurrent_gradient_metadata():
    class Step(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.wm_call = None

        def initial(self):
            return (torch.zeros(2, 1),) * 3

        def forward(self, state, previous, aux):
            return state + self.weight, previous + self.weight, aux + 1

    step = Step()
    count = [0]

    def candidate(state, previous, aux):
        if count[0]:
            assert state.requires_grad and previous.requires_grad
            assert not aux.requires_grad
        count[0] += 1
        return step(state, previous, aux)

    assert boundary_probe(step, candidate, 4)["first_failed_step"] is None
