"""Diagnostics preserve the real update; minimal controls retain physical BPTT."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch
from reflex.world_model.diagnostics_v31.attribution import summarize
from reflex.world_model.diagnostics_v31.full import check as full_check
from reflex.world_model.diagnostics_v31.full import scopes
from reflex.world_model.diagnostics_v31.minimal import ReflexStep, export
from reflex.world_model.diagnostics_v31.minimal import check as minimal_check
from reflex.world_model.diagnostics_v31.state import digest
from test_analytic_soft_termination import new_trainer
from test_compact_context import compact_trainer, fill


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


def fixture(tmp_path):
    source, row = compact_trainer(tmp_path)
    t = new_trainer(
        source,
        batch_imagination_amp=True,
        batch_imagination_amp_features=True,
        batch_imagination_critic=True,
        minimal_event_view=True,
        compact_event_inputs=True,
        reuse_post_event_fk=True,
        compact_event_pool_updates=True,
        collapse_full_event_subsets=True,
    )
    fill(t, row)
    t.cfg.horizon = 4
    t.cfg.actor_steps = 1
    t.cfg.gradient_audit_every = 0
    return t


def assert_tree_equal(a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_tree_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_full_actual_sampler_scopes_gradients_adam_and_rng_restore(tmp_path):
    t = fixture(tmp_path)
    t.update_actor()
    initial_updates = t.actor_updates
    initial = {
        name: copy.deepcopy(getattr(t, name).state_dict())
        for name in ("actor", "critic", "actor_opt", "critic_opt", "model")
    }
    rng = torch.get_rng_state().clone()
    sampler = t.sample
    scheduler = t.event_scheduler.rng.clone()
    report = full_check(t, tmp_path / "full", repeats=2, warmups=1)
    assert report["completed"] and report["numeric_passed"], report
    assert report["scope_semantics"]["passed"]
    assert report["sample_sha256"] == report["profiled_sample_sha256"]
    assert report["model_unchanged"] and report["actor_restored"]
    assert t.sample == sampler and t.actor_updates == initial_updates
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    torch.testing.assert_close(t.event_scheduler.rng, scheduler, atol=0, rtol=0)
    for name in ("actor", "critic", "model"):
        assert digest(getattr(t, name).state_dict()) == digest(initial[name])
    for name in ("actor_opt", "critic_opt"):
        assert_tree_equal(getattr(t, name).state_dict(), initial[name])
    keys = report["attribution"]["groups"]
    assert any("amp_features_batched" in k for k in keys)
    assert any("physical_wm" in k for k in keys)
    assert "backward/physical_wm" in keys


def test_scope_failure_restores_descriptors_and_compiled_handles(tmp_path):
    t = fixture(tmp_path)
    original = t._dynamics
    callback = t.event_scheduler._view
    with pytest.raises(RuntimeError, match="probe"), scopes(t):
        # The minimal-view identity guard must not be disturbed by instrumentation.
        assert t.event_scheduler._view == callback
        raise RuntimeError("probe")
    assert t._dynamics == original
    assert "forward" not in t.actor.__dict__


def test_capsule_roundtrip_and_minimal_h32_frozen_model_adam(tmp_path):
    t = fixture(tmp_path)
    with torch.no_grad():
        t.model.net[-1].weight.normal_(std=0.003)
    model = digest(t.model.state_dict())
    actor = digest(t.actor.state_dict())
    rng = torch.get_rng_state().clone()
    path = tmp_path / "capsule.pt"
    metadata = export(t, path)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    capsule = torch.load(path, weights_only=True)
    step = ReflexStep(capsule, "cpu")
    assert digest(step.model.state_dict()) == model
    assert metadata["source_model_sha256"] == model
    assert metadata["source_actor_sha256"] == actor
    torch.testing.assert_close(step.data["obs0"], capsule["inputs"]["obs0"])
    x = step.data
    raw = t.actor(t.obs_norm(x["obs0"]))
    target, _ = t.action_map(
        raw,
        x["blend"],
        x["launched"],
        x["scale"],
        x["offset"],
        x["down"],
        x["up"],
        x["limits"],
    )
    expected = t.model.predict_physical(x["state"], target, x["context"], x["pre"])
    torch.testing.assert_close(step(*step.initial())[0], expected)
    report = minimal_check(
        lambda: ReflexStep(capsule, "cpu"),
        tmp_path / "minimal",
        horizon=32,
        repeats=1,
        warmups=1,
        backend="aot_eager",
    )
    assert report["completed"] and report["numeric_passed"], report
    assert all(
        v["wm_frozen"] and v["physical_bptt_nonzero"]
        for v in report["drivers"].values()
    )
    assert digest(t.actor.state_dict()) == actor
    assert digest(t.model.state_dict()) == model


def event(name, thread=1, seq=-1, scope=0, parent=None, fwd=None, kernels=()):
    return NS(
        name=name,
        thread=thread,
        sequence_nr=seq,
        scope=scope,
        cpu_parent=parent,
        fwd_thread=fwd,
        device_type="CPU",
        is_async=False,
        self_cpu_time_total=1.0,
        kernels=kernels,
    )


def test_backward_sequence_owner_and_unmatched_work_are_not_guessed():
    root = event("v31/physical_wm")
    forward = event("CompiledFunction", seq=7, parent=root)
    outer = event("v31/complete_update")
    backward = event("CompiledFunctionBackward", seq=7, scope=1, fwd=1, parent=outer)
    kernel = event("aten::mm", parent=backward, kernels=[NS(duration=3)])
    unmatched = event(
        "OtherBackward", seq=99, scope=1, fwd=1, parent=outer, kernels=[NS(duration=4)]
    )
    engine = event(
        "autograd::engine::evaluate_function: CompiledFunctionBackward",
        seq=7,
        parent=outer,
        fwd=1,
    )
    result = summarize([root, forward, outer, engine, backward, kernel, unmatched])
    assert result["groups"]["backward/physical_wm"]["kernel_work_us"] == 3
    assert result["groups"]["backward/unmatched_or_shared"]["kernel_work_us"] == 4


def test_packed_compiled_path_scopes_remain_equivalent(tmp_path):
    source, row = compact_trainer(tmp_path)
    t = new_trainer(
        source,
        compile_dynamics=True,
        compile_fk=True,
        compile_v24_step=True,
        v24_fusion_scope="packed_heads",
        compile_backend="aot_eager",
        batch_imagination_amp=True,
        batch_imagination_amp_features=True,
        batch_imagination_critic=True,
        minimal_event_view=True,
        compact_event_inputs=True,
        reuse_post_event_fk=True,
        compact_event_pool_updates=True,
        collapse_full_event_subsets=True,
    )
    fill(t, row)
    t.cfg.horizon = 3
    t.cfg.actor_steps = 1
    t.cfg.gradient_audit_every = 0
    report = full_check(t, tmp_path / "compiled", repeats=1, warmups=1)
    assert report["completed"] and report["numeric_passed"]
    assert report["capture_excludes_compilation"], report["capture_compile_delta"]
    assert report["sampling_audit"]["input_hash_equal"]
    assert not hasattr(t, "_v31_regions")


def test_timeline_union_does_not_sum_overlapping_streams():
    from reflex.world_model.diagnostics_v31.attribution import activity_timeline

    root = event("v31/complete_update")
    root.time_range = NS(start=0, end=10)
    a, b = event("kernel_a"), event("kernel_b")
    for e, start, end in ((a, 0, 5), (b, 3, 9)):
        e.device_type = "CUDA"
        e.time_range = NS(start=start, end=end)
    result = activity_timeline([root, a, b])
    assert result["device_activity_union_us"] == 9
    assert result["unobserved_device_gap_us"] == 1
    annotation = event("v31/complete_update")
    annotation.device_type = "CUDA"
    annotation.time_range = NS(start=0, end=10)
    annotation.is_user_annotation = True
    assert activity_timeline([root, a, b, annotation]) == result
    from reflex.world_model.diagnostics_v31.trace_analysis import analyze

    trace = [
        {
            "ph": "X",
            "cat": "user_annotation",
            "name": "v31/complete_update",
            "ts": 0,
            "dur": 10,
        },
        {
            "ph": "X",
            "cat": "gpu_user_annotation",
            "name": "v31/complete_update",
            "ts": 0,
            "dur": 10,
        },
        {"ph": "X", "cat": "kernel", "name": "a", "ts": 0, "dur": 5},
        {"ph": "X", "cat": "kernel", "name": "b", "ts": 3, "dur": 6},
        {
            "ph": "X",
            "cat": "gpu_memcpy",
            "name": "DtoH",
            "ts": 2,
            "dur": 1,
            "args": {"bytes": 4},
        },
    ]
    summary = analyze(trace)
    assert summary["device_activity_union_us"] == 9
    assert summary["kernel_count"] == 2
    assert summary["memcpy_bytes"] == {"DtoH": 4}


def test_native_adapter_interface_with_fixture_checkout(tmp_path, monkeypatch):
    import importlib
    import subprocess
    import sys

    from reflex.world_model.diagnostics_v31.tracker_native import factory_from_repo

    repo = tmp_path / "tracker"
    (repo / "source/trackerLab").mkdir(parents=True)
    path = repo / "source/trackerLab/networks.py"
    source = """import torch
from torch import nn
AMP_DTYPE = torch.bfloat16
class WorldModel(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.proj=nn.Linear(num_joints,num_joints)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def step(self,S,a,xi):
        p,R,v,w,q,dq=S
        return p+.02*v,R,v,w,q+.01*self.proj(a),dq+.01*a
class Policy(nn.Module):
    def __init__(self, obs_dim, goal_dim, hist, num_joints):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim*hist+goal_dim,8),nn.SiLU(),nn.Linear(8,num_joints))
    def forward(self,h,g):
        return self.net(torch.cat((h.flatten(1),g),-1))
"""
    path.write_text(source)
    module_spec = importlib.util.spec_from_file_location("v31_fixture_networks", path)
    networks = importlib.util.module_from_spec(module_spec)
    monkeypatch.setitem(sys.modules, networks.__name__, networks)
    module_spec.loader.exec_module(networks)
    spec = NS(num_joints=29, goal_frame_dim=6, context_dim=33, default_q=[0.0] * 29)
    modules = {
        "trackerLab.algorithms.supermimic.networks": networks,
        "trackerLab.tracker_env.mdp.supermimic": NS(normalize_xi=lambda x: x),
        "trackerLab.kinematics.supermimic.rotations": NS(
            so3_exp=lambda x: torch.eye(3).expand(len(x), -1, -1).clone()
        ),
        "trackerTask.supermimic.robots": NS(get_robot=lambda _: spec),
    }
    original_import, original_run = importlib.import_module, subprocess.run
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a, **kw: (
            modules[name] if name in modules else original_import(name, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kw: (
            NS(stdout="fixture-revision\n")
            if args[:1] == ["git"]
            else original_run(args, **kw)
        ),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("SUPERMIMIC_REPO_ROOT", str(repo))
    factory, metadata = factory_from_repo(repo, "cpu", 2)
    assert factory().perturbed_heads == 1
    report = minimal_check(
        factory,
        tmp_path / "native",
        device="cpu",
        horizon=4,
        repeats=1,
        warmups=1,
        backend="aot_eager",
        metadata=metadata,
    )
    assert report["completed"] and report["numeric_passed"]
    assert all(v["physical_bptt_nonzero"] for v in report["drivers"].values())

    def no_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0], stderr="not a git repository")

    monkeypatch.setattr(subprocess, "run", no_git)
    _, no_git_metadata = factory_from_repo(repo, "cpu", 2)
    assert no_git_metadata["revision"] is None
    assert no_git_metadata["revision_unavailable_reason"] == "not a git repository"
