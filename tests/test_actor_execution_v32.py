"""CPU equivalence and rollback checks; no GPU speed claims."""

import copy
import types
from types import SimpleNamespace as NS

import pytest
import torch
from reflex.world_model.actor_execution_v32 import (
    MINIMAL_EVENT_FIELDS,
    gather_event_inputs,
    packed_pose_carry,
)
from reflex.world_model.actor_execution_v32_check import check
from reflex.world_model.diagnostics_v31.state import Snapshot, digest
from reflex.world_model.events_v24.fast_view import build
from reflex.world_model.events_v24.runtime import EventTerm
from reflex.world_model.joint import returns
from reflex.world_model.trainer import freeze
from test_actor_diagnostics_v31 import assert_tree_equal
from test_actor_diagnostics_v31 import fixture as trainer_fixture
from test_actor_execution_v30 import (
    equal_constant_pose_event,
    perturb_all,
    robot_joint_event,
    robot_velocity_event,
)
from test_analytic_soft_termination import new_trainer
from test_v24_events import setup, toy_control


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


@pytest.mark.parametrize("ids", [[0, 1, 2], [2, 0], []])
def test_minimal_gather_does_not_touch_unused_fields_or_poses(ids):
    scheduler = NS(pool_terms=[0])
    scheduler._view = types.MethodType(build, scheduler)
    batch = {k: torch.arange(9).reshape(3, 3) for k in MINIMAL_EVENT_FIELDS}
    batch["unused_context"] = torch.randn(3, 765)

    class NoIteration:
        def __iter__(self):
            raise AssertionError("Minimal event view must not read any FK pose")

    selected = torch.tensor(ids, dtype=torch.long)
    result, poses = gather_event_inputs(scheduler, batch, NoIteration(), selected, 3)
    assert poses == () and tuple(result) == MINIMAL_EVENT_FIELDS
    for key in result:
        torch.testing.assert_close(result[key], batch[key][selected], atol=0, rtol=0)
    scheduler.pool_terms = []
    batch.pop("event_agent")  # legal when no foreign-world pools exist
    result, _ = gather_event_inputs(scheduler, batch, NoIteration(), selected, 3)
    assert tuple(result) == MINIMAL_EVENT_FIELDS[:3]


def test_custom_view_falls_back_with_scalars_and_metadata():
    scheduler = NS(_view=lambda *args: None)
    metadata = object()
    batch = {"context": torch.randn(3, 4), "scalar": torch.tensor(3), "meta": metadata}
    poses = ({"a": torch.randn(3, 3)}, {"a": torch.randn(3, 4)})
    ids = torch.tensor([2, 0])
    result, actual = gather_event_inputs(scheduler, batch, poses, ids, 3)
    assert result["meta"] is metadata and result["scalar"] is batch["scalar"]
    torch.testing.assert_close(result["context"], batch["context"][ids], atol=0, rtol=0)
    for group, expected in zip(actual, poses):
        torch.testing.assert_close(group["a"], expected["a"][ids], atol=0, rtol=0)


@pytest.mark.parametrize(
    "active", [[True, True, True], [False, False, False], [True, False, True]]
)
def test_packed_pose_values_input_gradients_and_immutability(active):
    inputs = [
        torch.randn(3, 30, width, dtype=torch.float64, requires_grad=True)
        for width in (3, 4, 3, 4)
    ]
    names = [f"body{i}" for i in range(30)]
    old = tuple(dict(zip(names, x.unbind(1))) for x in inputs[:2])
    pred = tuple(dict(reversed(list(zip(names, x.unbind(1))))) for x in inputs[2:])
    originals = [x.detach().clone() for x in inputs]
    mask = torch.tensor(active)
    reference = tuple(
        {k: torch.where(mask[:, None], b[k], a[k]) for k in a}
        for a, b in zip(old, pred)
    )
    actual = packed_pose_carry(old, pred, mask)
    weight = torch.randn(3, 30, 7, dtype=torch.float64)

    def loss(poses):
        values = torch.cat([torch.stack(list(g.values()), 1) for g in poses], -1)
        return (values.square() * weight).sum()

    for a, b in zip(actual, reference):
        assert tuple(a) == tuple(b)
        for name in a:
            torch.testing.assert_close(a[name], b[name], atol=0, rtol=0)
    gradients = [torch.autograd.grad(loss(x), inputs) for x in (reference, actual)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(inputs, originals):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert packed_pose_carry(({},), ({},), mask) == ({},)


@pytest.mark.parametrize("opt", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize(
    "callback",
    [toy_control, robot_velocity_event, equal_constant_pose_event, robot_joint_event],
)
def test_h32_events_raw_gradient_and_adam_equivalence(
    tmp_path, opt, truncate, callback
):
    t = trainer_fixture(tmp_path)
    t.cfg.horizon = 32
    t.event_scheduler.terms[0].function = callback
    # Both waiting->launch and partial valid subsets; no filter on launched.
    start = t.sample(starts=True, n=4)
    start["episode_steps"][:] = 0
    if truncate:
        start["episode_steps"][0] = t.task.max_steps - 4
    snapshot = Snapshot(t)
    rows = []
    for flags in ((False, False), opt):
        snapshot.reset()
        t.cfg.compact_actor_event_inputs, t.cfg.packed_pose_carry = flags
        with freeze(t.model, t.termination, t.prior._disc, t.critic):
            imagined = t.imagine({k: v.clone() for k, v in start.items()}, trace=True)
            loss = -returns(imagined, t.critic, t.cfg.gamma).mean()
            t.actor_opt.zero_grad(set_to_none=True)
            loss.backward()
            raw_gradient = torch.cat(
                [p.grad.detach().flatten().clone() for p in t.actor.parameters()]
            )
            assert torch.isfinite(raw_gradient).all() and raw_gradient.norm() > 0
            torch.nn.utils.clip_grad_norm_(t.actor.parameters(), 1.0)
            t.actor_opt.step()
        rows.append(
            (
                {k: v.detach().clone() for k, v in imagined.items()},
                raw_gradient,
                copy.deepcopy(t.actor.state_dict()),
                copy.deepcopy(t.actor_opt.state_dict()),
            )
        )
        assert imagined["predicted_launch_probability"].any()
        if truncate:
            assert imagined["truncated"].any()
    for key in rows[0][0]:
        torch.testing.assert_close(rows[0][0][key], rows[1][0][key], atol=0, rtol=0)
    torch.testing.assert_close(rows[0][1], rows[1][1])
    for key in rows[0][2]:
        torch.testing.assert_close(rows[0][2][key], rows[1][2][key])
    assert rows[0][3]["param_groups"] == rows[1][3]["param_groups"]
    for i, state in rows[0][3]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, rows[1][3]["state"][i][key])


def test_foreign_timer_and_position_dependent_write_gradients(tmp_path):
    t, b, runtime, event = setup(tmp_path)
    runtime._view = types.MethodType(build, runtime)
    runtime.terms = [EventTerm("perturb", perturb_all, {}, (0.5, 1.0), False, True)]
    runtime.pool_terms, runtime.pool_index = [0], {0: 0}
    pools = torch.ones(4, 1, 4)
    pools[:, 0, 1] = 0  # source 1 fires; sampled own sources are 0 and 2
    ids = torch.tensor([0, 2])
    outputs = []
    for compact in (False, True):
        physical = b["state"].clone().requires_grad_()
        poses = t.bridge.fk(physical)
        batch, pp = (
            gather_event_inputs(runtime, b, poses, ids, 4)
            if compact
            else (
                {
                    k: v[ids] if torch.is_tensor(v) and v.ndim and len(v) == 4 else v
                    for k, v in b.items()
                },
                tuple({k: v[ids] for k, v in g.items()} for g in poses),
            )
        )
        torch.manual_seed(932)
        plan, after, pool = runtime.step(
            batch,
            physical[ids],
            pp,
            event[ids].clone(),
            pools[ids],
            torch.ones(2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            isolated_rng=False,
        )
        grad = torch.autograd.grad(plan.targets.sum(), physical)[0]
        assert grad[ids, :3].abs().sum() > 0
        outputs.append(
            (plan.targets.detach(), plan.mask, after, pool, grad, torch.get_rng_state())
        )
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_paired_checker_preserves_existing_adam_rng_and_flags(tmp_path):
    t = trainer_fixture(tmp_path)
    t.update_actor()  # nonempty Adam state
    initial = Snapshot(t)
    t.cfg.compact_actor_event_inputs = True
    report = check(t, tmp_path / "check", repeats=2, warmups=1)
    assert report["completed"] and report["numeric_passed"], report
    assert report["actor_restored"] and report["model_unchanged"]
    assert t.cfg.compact_actor_event_inputs and not t.cfg.packed_pose_carry
    for name in initial.names:
        assert_tree_equal(getattr(t, name).state_dict(), initial.weights[name])
    torch.testing.assert_close(torch.get_rng_state(), initial.rng[2], atol=0, rtol=0)


def test_checkpoint_execution_flags_do_not_migrate_weights_or_adam(tmp_path):
    t = trainer_fixture(tmp_path)
    t.update_actor()
    path = tmp_path / "base.pt"
    t.save(path)
    legacy = torch.load(path, weights_only=False)
    legacy["config"].pop("compact_actor_event_inputs")
    legacy["config"].pop("packed_pose_carry")
    torch.save(legacy, path)
    before = digest(t.actor.state_dict())
    adam = copy.deepcopy(t.actor_opt.state_dict())
    t.cfg.compact_actor_event_inputs = t.cfg.packed_pose_carry = True
    t.load(path)
    assert digest(t.actor.state_dict()) == before
    assert_tree_equal(t.actor_opt.state_dict(), adam)


def test_paired_compiled_heads_cover_new_pose_strides(tmp_path):
    source = trainer_fixture(tmp_path)
    t = new_trainer(
        source,
        compile_dynamics=True,
        compile_fk=True,
        compile_v24_step=True,
        v24_fusion_scope="packed_heads",
        compile_backend="aot_eager",
    )
    t.replay.load_state_dict(source.replay.state_dict())
    t.cfg.horizon = 3
    report = check(t, tmp_path / "compiled", repeats=1, warmups=2)
    assert report["completed"] and report["numeric_passed"], report
    assert report["performance_usable"], report["steady_compile_delta"]


def test_checker_failure_restores_flags_sampler_weights_and_rng(tmp_path, monkeypatch):
    t = trainer_fixture(tmp_path)
    sampler, actor, rng = t.sample, digest(t.actor.state_dict()), torch.get_rng_state()
    original = t.cfg.compact_actor_event_inputs, t.cfg.packed_pose_carry

    def fail():
        t.cfg.compact_actor_event_inputs = True
        torch.rand(2)
        with torch.no_grad():
            next(t.actor.parameters()).add_(1)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(t, "update_actor", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        check(t, tmp_path / "failed", repeats=1, warmups=1)
    assert (t.cfg.compact_actor_event_inputs, t.cfg.packed_pose_carry) == original
    assert t.sample == sampler and digest(t.actor.state_dict()) == actor
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
