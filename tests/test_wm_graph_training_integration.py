"""Persistent WM encoding and compiled training: no silent legacy conversion."""

import copy

import pytest
import torch
from reflex.world_model.diagnostics_v31.state import digest
from test_actor_diagnostics_v31 import assert_tree_equal, fixture
from test_analytic_soft_termination import new_trainer


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


def stable(source, **kwargs):
    t = new_trainer(
        source, wm_encoding="stable_heading_v1", compile_backend="aot_eager", **kwargs
    )
    t.prior = copy.deepcopy(source.prior)
    t.replay.load_state_dict(source.replay.state_dict())
    return t


def test_encoding_checkpoint_normalizers_optimizer_and_old_rejection(tmp_path):
    source = fixture(tmp_path)
    source.cfg.compile_backend = "aot_eager"
    source.update_actor()
    legacy = tmp_path / "legacy.pt"
    source.save(legacy)
    saved = torch.load(legacy, weights_only=False)
    saved["config"].pop("wm_encoding")
    saved["config"].pop("wm_training_compile_mode")
    torch.save(saved, legacy)
    # Existing checkpoints still load with legacy defaults unchanged.
    old = new_trainer(source, compile_backend="aot_eager")
    old.load(legacy)
    assert_tree_equal(old.actor_opt.state_dict(), source.actor_opt.state_dict())
    t = stable(source)
    before = digest(t.model.state_dict())
    with pytest.raises(ValueError, match="WM encoding mismatch"):
        t.load(legacy)
    assert digest(t.model.state_dict()) == before
    t.fit_model()
    t.update_actor()
    assert t.model.inputs.mean[3] == 0 and t.model.inputs.std[3] == 0.01
    path = tmp_path / "stable.pt"
    t.save(path)
    loaded = stable(source, compile_wm_rollout=True, compile_wm_training_step=True)
    pointers = (
        loaded.model.inputs.mean.data_ptr(),
        loaded.model.inputs.std.data_ptr(),
        loaded.model.net[0].weight.data_ptr(),
    )
    loaded.load(path)
    assert pointers == (
        loaded.model.inputs.mean.data_ptr(),
        loaded.model.inputs.std.data_ptr(),
        loaded.model.net[0].weight.data_ptr(),
    )
    assert loaded.learning_contract["wm_encoding"]["name"] == "stable_heading_v1"
    assert_tree_equal(loaded.model.state_dict(), t.model.state_dict())
    assert_tree_equal(loaded.model_opt.state_dict(), t.model_opt.state_dict())
    assert_tree_equal(loaded.actor_opt.state_dict(), t.actor_opt.state_dict())
    with pytest.raises(ValueError, match="WM encoding mismatch"):
        old.load(path)


def test_compiled_h8_then_actor_continuous_updates_and_resume(tmp_path):
    source = fixture(tmp_path)
    initial_rng = torch.get_rng_state().clone()
    trainers = []
    for compiled in (False, True):
        torch.set_rng_state(initial_rng)
        t = stable(
            source, compile_wm_rollout=compiled, compile_wm_training_step=compiled
        )
        t.cfg.model_steps, t.cfg.model_window = 2, 8
        t.cfg.horizon = 16
        trainers.append(t)
    for iteration in range(3):
        rng = torch.get_rng_state().clone()
        for t in trainers:
            torch.set_rng_state(rng)
            t.fit_model()
            t.update_actor()
        for name in ("model", "actor", "critic"):
            for key, value in getattr(trainers[0], name).state_dict().items():
                torch.testing.assert_close(
                    value, getattr(trainers[1], name).state_dict()[key]
                )
    path = tmp_path / "graph.pt"
    trainers[1].save(path)
    resumed = stable(source, compile_wm_rollout=True, compile_wm_training_step=True)
    resumed.cfg.model_steps, resumed.cfg.model_window, resumed.cfg.horizon = 2, 8, 16
    resumed.load(path)
    assert resumed.actor_updates == trainers[1].actor_updates
    assert_tree_equal(
        resumed.model_opt.state_dict(), trainers[1].model_opt.state_dict()
    )


def test_training_graph_mode_does_not_enable_actor_graph(tmp_path, monkeypatch):
    calls = []

    def compile_fn(function, **kwargs):
        calls.append((getattr(function, "__name__", ""), kwargs))
        return function

    monkeypatch.setattr(torch, "compile", compile_fn)
    source = fixture(tmp_path)
    new_trainer(
        source,
        wm_encoding="stable_heading_v1",
        compile_wm_rollout=True,
        compile_wm_training_step=True,
        wm_training_compile_mode="reduce-overhead",
        wm_compile_mode="default",
        compile_dynamics=True,
        compile_backend="inductor",
    )
    for name, kwargs in calls:
        if name in ("v30_model_rollout", "v24_model_training_step"):
            assert kwargs["mode"] == "reduce-overhead"
        elif name == "predict_physical":
            assert kwargs.get("mode", "default") == "default"


def test_joint_acceptance_restores_all_state_on_cpu(tmp_path):
    from reflex.world_model.wm_graph_joint_check import check

    source = fixture(tmp_path)
    t = stable(source, compile_wm_rollout=True, compile_wm_training_step=True)
    t.cfg.model_steps, t.cfg.model_window, t.cfg.horizon = 2, 8, 16
    t.fit_normalizers()
    weights = copy.deepcopy(t.model.state_dict())
    actor = copy.deepcopy(t.actor.state_dict())
    amp = copy.deepcopy(t.prior.amp_state_dict())
    t.prior._prev_valid.fill_(True)
    prior_previous = t.prior._prev.clone(), t.prior._prev_valid.clone()
    rng = torch.get_rng_state().clone()
    t.wm_warmup_inherited = False
    t.cfg.wm_only_rounds = 500
    result = check(t, tmp_path, repeats=2, warmups=1)
    assert result["completed"] and result["numeric_passed"], result
    assert result["continuous_update_numeric_passed"]
    assert result["model_restored"] and result["actor_restored"]
    assert not t.wm_warmup_inherited
    assert_tree_equal(t.model.state_dict(), weights)
    assert_tree_equal(t.actor.state_dict(), actor)
    import numpy as np

    def arrays_as_lists(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: arrays_as_lists(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [arrays_as_lists(v) for v in value]
        return value

    assert_tree_equal(arrays_as_lists(t.prior.amp_state_dict()), arrays_as_lists(amp))
    assert_tree_equal(torch.get_rng_state(), rng)
    assert_tree_equal((t.prior._prev, t.prior._prev_valid), prior_previous)


def test_stable_minimal_capsule_keeps_encoding(tmp_path):
    from reflex.world_model.diagnostics_v31.minimal import ReflexStep, export

    source = fixture(tmp_path)
    t = stable(source)
    t.fit_normalizers()
    path = tmp_path / "capsule.pt"
    export(t, path)
    step = ReflexStep(torch.load(path, weights_only=True), "cpu")
    assert step.model.encoding == "stable_heading_v1"
    data = step.data
    expected = t.model.features(
        data["state"],
        torch.zeros_like(data["previous_action"]),
        data["context"],
        data["pre"],
    )
    actual = step.model.features(
        data["state"],
        torch.zeros_like(data["previous_action"]),
        data["context"],
        data["pre"],
    )
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
