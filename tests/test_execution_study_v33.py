"""Full-path diagnostic equivalence, source-site coverage and exception safety."""

import copy
import gc

import pytest
import torch
from reflex.world_model.diagnostics_v31.state import Snapshot, digest
from reflex.world_model.execution_study_v33 import (
    CALLBACK,
    PREFIX,
    check,
    detailed_scopes,
    phase_check,
    source_sites,
)
from test_actor_diagnostics_v31 import assert_tree_equal
from test_actor_diagnostics_v31 import fixture as base_fixture
from test_analytic_soft_termination import new_trainer


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


def fixture(tmp_path):
    t = base_fixture(tmp_path)
    t.cfg.compact_actor_event_inputs = t.cfg.packed_pose_carry = True
    return t


def test_source_sites_values_gradients_lines_and_exception_restore():
    names = (
        "nonzero",
        "item",
        "__bool__",
        "__int__",
        "__float__",
        "__getitem__",
        "__setitem__",
    )
    before = {n: getattr(torch.Tensor, n) for n in names}

    def compute():
        x = torch.tensor([-1.0, 2.0, 3.0], requires_grad=True)
        mask = x > 0
        ids = mask.nonzero().flatten()
        y = x.clone()
        y[mask] = x[mask] * 2
        values = (
            bool(mask.any()),
            int(ids[0]),
            float(x.detach()[1]),
            x.detach()[1].item(),
        )
        return y, torch.autograd.grad(y.square().sum(), x)[0], values

    reference = compute()
    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profiler,
        source_sites(include_cpu=True),
    ):
        actual = compute()
    for a, b in zip(reference[:2], actual[:2]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert actual[2] == reference[2]
    labels = [e.name for e in profiler.events() if e.name.startswith(PREFIX)]
    assert any("nonzero|" in n and "test_execution_study_v33.py:" in n for n in labels)
    assert any("__getitem__|" in n for n in labels)
    with pytest.raises(RuntimeError, match="probe"), source_sites(include_cpu=True):
        raise RuntimeError("probe")
    assert all(getattr(torch.Tensor, n) is f for n, f in before.items())


def test_detailed_scopes_keep_view_identity_and_restore_on_failure(tmp_path):
    t = fixture(tmp_path)
    view, step, invoke = (
        t.event_scheduler._view,
        t.event_scheduler.step,
        t.event_scheduler._invoke,
    )
    with pytest.raises(RuntimeError, match="probe"), detailed_scopes(t):
        assert t.event_scheduler._view == view
        raise RuntimeError("probe")
    assert t.event_scheduler._view == view and t.event_scheduler.step == step
    assert t.event_scheduler._invoke == invoke
    assert not hasattr(t.event_scheduler, "_v33_regions")


def test_coarse_phase_checks_preserve_adam_rng_and_gc_callbacks(tmp_path):
    t = fixture(tmp_path)
    t.update_actor()
    initial, callbacks = Snapshot(t), list(gc.callbacks)
    r = phase_check(t, tmp_path / "phases.json", repeats=2)
    assert r["completed"] and r["numeric_passed"]
    assert not any(r["compile_delta"].values())
    assert gc.callbacks == callbacks
    assert "_v33_phase_recorder" not in t.__dict__
    assert {p["name"] for p in r["recorded"][0]["phases"]} == {
        "actor_forward",
        "actor_backward",
        "actor_clip_adam",
    }
    for name in initial.names:
        assert_tree_equal(getattr(t, name).state_dict(), initial.weights[name])
    torch.testing.assert_close(torch.get_rng_state(), initial.rng[2], rtol=0, atol=0)


def test_phase_failure_removes_hooks_and_restores_state(tmp_path, monkeypatch):
    t = fixture(tmp_path)
    before, callbacks = digest(t.actor.state_dict()), list(gc.callbacks)
    original = t.update_actor
    count = 0

    def fail():
        nonlocal count
        count += 1
        if count == 4:  # recorded measurement after both warmups
            raise RuntimeError("recorded failure")
        return original()

    monkeypatch.setattr(t, "update_actor", fail)
    with pytest.raises(RuntimeError, match="recorded failure"):
        phase_check(t, tmp_path / "failed.json", repeats=1)
    assert gc.callbacks == callbacks
    assert digest(t.actor.state_dict()) == before
    assert "_v33_phase_recorder" not in t.__dict__


def test_full_native_site_profile_and_actual_parameter_updates(tmp_path):
    t = fixture(tmp_path)
    original = copy.deepcopy(t.actor_opt.state_dict())
    r = check(t, tmp_path / "study", repeats=1, warmups=1)
    assert r["completed"] and r["numeric_passed"] and r["attribution_usable"]
    import json

    report = json.loads((tmp_path / "study/full/actor_full_v31.json").read_text())
    assert report["scope_semantics"]["passed"]
    assert report["profiled_parameter_and_gradient_equivalence"]["passed"]
    groups = report["execution_sites"]["groups"]
    assert any(k.startswith(PREFIX) for k in groups)
    assert any(k.startswith(CALLBACK) for k in groups)
    assert sum(v["nonzero_calls"] for v in groups.values()) > 0
    assert_tree_equal(t.actor_opt.state_dict(), original)


def test_compiled_plant_is_not_replaced_by_source_site_hooks(tmp_path):
    source = fixture(tmp_path)
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
    r = check(t, tmp_path / "compiled", repeats=1, warmups=2)
    assert r["completed"] and r["numeric_passed"]
    assert r["attribution_usable"] and r["timings_usable"], r
