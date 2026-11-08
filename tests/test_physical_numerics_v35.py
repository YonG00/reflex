"""CPU capture/value/VJP/Adam checks, never CUDA or performance claims."""

import pytest
import torch
from reflex.world_model.diagnostics_v31.minimal import ReflexStep, check, export
from reflex.world_model.diagnostics_v31.state import digest
from reflex.world_model.physical_compile_v35 import (
    DRIVERS,
    build,
    components,
    update_drift,
)
from reflex.world_model.physical_numerics_v35 import stage_probe
from test_actor_diagnostics_v31 import fixture as trainer_fixture


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


def make_step(tmp_path):
    t = trainer_fixture(tmp_path)
    with torch.no_grad():
        t.model.net[-1].weight.normal_(std=0.003)
    path = tmp_path / "capsule.pt"
    export(t, path)
    capsule = torch.load(path, weights_only=True)
    return ReflexStep(capsule, "cpu"), capsule


@pytest.mark.parametrize("driver", DRIVERS)
def test_boundary_program_values_gradients_and_model_identity(tmp_path, driver):
    step, _ = make_step(tmp_path)
    model = step.model
    before = digest(model.state_dict())
    d = step.data
    target = torch.randn(len(d["state"]), 29)
    rows, gradients = [], []
    for call in [model.predict_physical, build(model, driver, "aot_eager")]:
        args = [
            v.detach().clone().requires_grad_()
            for v in (d["state"], target, d["context"], d["pre"])
        ]
        result = call(*args)
        rows.append(result)
        grads = torch.autograd.grad(result.square().mean(), args, allow_unused=True)
        gradients.append(
            [torch.zeros_like(a) if g is None else g for a, g in zip(args, grads)]
        )
    torch.testing.assert_close(*rows)
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b)
    assert digest(model.state_dict()) == before
    assert not any(k.startswith("_orig_mod") for k in model.state_dict())


def test_same_input_components_and_vjp_are_separate_from_rollout(tmp_path):
    step, _ = make_step(tmp_path)
    before = digest(step.state_dict())
    result = stage_probe(step, steps=(1, 6), backend="aot_eager")
    assert result["completed"]
    assert set(components(step.model)) == {
        "features",
        "normalization",
        "mlp",
        "integrate",
        "encoder",
    }
    for row in result["rows"]:
        for report in row["components"].values():
            assert report["value"]["passed"] and report["input_vjp"]["passed"], report
    assert digest(step.state_dict()) == before
    assert all(p.grad is None for p in step.parameters())


def test_h32_boundaries_actual_adam_and_bptt(tmp_path):
    _step, capsule = make_step(tmp_path)
    result = check(
        lambda: ReflexStep(capsule, "cpu"),
        tmp_path / "h32",
        horizon=32,
        repeats=1,
        warmups=1,
        backend="aot_eager",
        drivers=(
            "eager",
            "segmented",
            "wm_mlp",
            "wm_layerwise",
            "wm_mlp_integrate",
            "wm_encoder_mlp",
        ),
    )
    assert result["completed"] and result["numeric_passed"], result
    for entry in result["drivers"].values():
        assert entry["physical_bptt_nonzero"]
        assert entry["boundary_probe"]["first_failed_step"] is None


def test_parameter_drift_reports_gradient_at_same_coordinate():
    ref = {
        "parameter/x": torch.tensor([1.0, 1.0]),
        "gradient/0": torch.tensor([0.0, 1e-10]),
    }
    actual = {
        "parameter/x": torch.tensor([1.0, 1.00002]),
        "gradient/0": torch.tensor([0.0, -1e-10]),
    }
    row = update_drift(ref, actual, [("x", None)])[0]
    assert row["flat_index"] == 1
    assert row["coordinate"] == [1]
    assert row["reference_applied_gradient"] > 0 > row["candidate_applied_gradient"]
    assert row["max_abs_update_difference"] > 0


def test_optional_layer_probes_and_progress_are_explicit(tmp_path):
    step, _ = make_step(tmp_path)
    progress = []
    report = stage_probe(
        step,
        steps=(1,),
        backend="aot_eager",
        progress=lambda value: progress.append(len(value["rows"][-1]["components"])),
        mlp_layers=True,
    )
    expected_count = 5 + len(step.model.net)
    assert progress == list(range(1, expected_count + 1))
    row = report["rows"][0]
    assert len(row["components"]) == expected_count
    assert all(
        v["value"]["passed"] and v["input_vjp"]["passed"]
        for v in row["components"].values()
    )


def test_missing_com_uses_original_integrator(tmp_path):
    step, _ = make_step(tmp_path)
    step.model.com_indices = None
    d = step.data
    args = (d["state"], torch.ones(len(d["state"]), 29), d["context"], d["pre"])
    expected = step.model.predict_physical(*args)
    actual = build(step.model, "wm_mlp", "aot_eager")(*args)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("driver", ["wm_mlp", "wm_layerwise"])
def test_recorded_h8_wm_supervision_gradient_and_actual_adam(tmp_path, driver):
    import copy

    t = trainer_fixture(tmp_path)
    t.cfg.model_window = 8
    with torch.no_grad():
        t.model.net[-1].weight.normal_(std=0.003)
    seq = t.sample(8)
    initial = copy.deepcopy(t.model.state_dict())
    optimizer = copy.deepcopy(t.model_opt.state_dict())
    original = t.model.predict_physical
    results = []
    try:
        for selected in [None, driver]:
            t.model.load_state_dict(initial)
            t.model_opt.load_state_dict(copy.deepcopy(optimizer))
            t.model_opt.zero_grad(set_to_none=True)
            t.model.predict_physical = (
                original if selected is None else build(t.model, selected, "aot_eager")
            )
            post, physical_loss, term_loss, _ = t.v30_model_rollout(seq)
            losses = physical_loss + t.cfg.term_loss_weight * term_loss
            mask = seq.get("loss_mask")
            loss = (
                losses.mean()
                if mask is None
                else (
                    torch.where(mask, losses, 0).sum(0) / mask.sum(0).clamp_min(1)
                ).mean()
            )
            loss.backward()
            grads = [p.grad.detach().clone() for p in t.model.parameters()]
            assert any(g.norm() > 0 for g in grads)
            torch.nn.utils.clip_grad_norm_(
                t.model.parameters(), t.cfg.model_grad_clip, error_if_nonfinite=True
            )
            t.model_opt.step()
            results.append(
                (
                    post.detach().clone(),
                    loss.detach().clone(),
                    grads,
                    copy.deepcopy(t.model.state_dict()),
                )
            )
        torch.testing.assert_close(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])
        for a, b in zip(results[0][2], results[1][2]):
            torch.testing.assert_close(a, b)
        for key in results[0][3]:
            torch.testing.assert_close(results[0][3][key], results[1][3][key])
    finally:
        del t.model.predict_physical


def test_physical_trial_cannot_use_itself_as_reference(tmp_path):
    with pytest.raises(ValueError, match="eager as first"):
        check(lambda: None, tmp_path / "invalid", drivers=("wm_mlp",))
