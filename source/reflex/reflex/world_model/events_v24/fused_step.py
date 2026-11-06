"""Full-graph capture, numerical equivalence, and warm H32 timing for V24."""

import json
import time
from pathlib import Path

import torch

from ..joint import returns
from ..trainer import freeze
from .compare import seed


def check(t, output, repeats=3):
    if t._v24_step is None:
        raise ValueError("Use compile_v24_step=true candidate config")
    out = Path(output)
    compiled_step, compiled_obs = t._v24_step, t._observations
    event_rng = t.event_scheduler.rng.clone()
    counters = t.actor_updates, t.model_updates
    result = {
        "completed": False,
        "optimizer_updates": 0,
        "batch": t.cfg.actor_batch,
        "horizon": t.cfg.horizon,
        "tf32": t.cfg.tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "fusion_scope": t.cfg.v24_fusion_scope,
    }

    def progress(stage):
        result["stage"] = stage
        (out / "fusion_check.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"fusion_stage": stage}), flush=True)

    def sync():
        if str(t.device).startswith("cuda"):
            torch.cuda.synchronize()

    def run(b):
        seed(10601)
        t.event_scheduler.rng = event_rng.clone()
        with freeze(t.model, t.termination, t.prior._disc, t.critic):
            v = t.imagine(start_batch=b)
            loss = -returns(v, t.critic, t.cfg.gamma).mean() / t.cfg.horizon
            gradients = torch.autograd.grad(loss, tuple(t.actor.parameters()))
        return {
            k: v[k].detach()
            for k in (
                "obs",
                "critic_obs",
                "reward",
                "done",
                "truncated",
                "predicted_launch_probability",
            )
        }, tuple(g.detach() for g in gradients)

    def compare(expected, actual):
        values, grads = expected
        other, og = actual
        result.setdefault("comparison_errors", {})[result["stage"]] = {
            **{
                key: float((values[key].float() - other[key].float()).abs().max())
                for key in values
            },
            "actor_gradient": max(
                float((x - y).abs().max()) for x, y in zip(grads, og)
            ),
        }
        (out / "fusion_check.json").write_text(json.dumps(result, indent=2))
        for key in values:
            torch.testing.assert_close(
                values[key],
                other[key],
                atol=2e-4,
                rtol=3e-3,
                msg=lambda x, key=key: key + ": " + x,
            )
        for x, y in zip(grads, og):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=3e-3)
        return max(float((x - y).abs().max()) for x, y in zip(grads, og))

    try:
        seed(10600)
        small = t.sample(starts=True, n=8)
        t._v24_step, t._observations = None, t.imagined_observations
        progress("baseline_small")
        from ..diagnostics import eager_diagnostic

        with eager_diagnostic(t):
            expected = run(small)
        # Prove Torch 2.7 can capture all active task branches before codegen.
        t._v24_step = torch.compile(
            t.v24_physical_step, fullgraph=True, backend="eager", dynamic=False
        )
        t._observations = torch.compile(
            t.imagined_observations, fullgraph=True, backend="eager", dynamic=False
        )
        progress("fullgraph_capture")
        normal_fk, normal_post = t._fk, t._post
        t._fk, t._post = t.bridge.fk, t.model.predict_post
        try:
            result["capture_gradient_max_abs_error"] = compare(expected, run(small))
        finally:
            t._fk, t._post = normal_fk, normal_post
        t._v24_step, t._observations = compiled_step, compiled_obs
        if t.cfg.actor_batch < 512:
            t._v24_step, t._observations = None, t.imagined_observations
            inductor_reference = run(small)
            t._v24_step, t._observations = compiled_step, compiled_obs
            progress("inductor_small_compile")
            result["small_gradient_max_abs_error"] = compare(
                inductor_reference, run(small)
            )
        else:
            result["inductor_validation_batch"] = t.cfg.actor_batch
        del expected
        seed(10602)
        batch = t.sample(starts=True, n=t.cfg.actor_batch)
        for variant in ("baseline", "fused"):
            t._v24_step = compiled_step if variant == "fused" else None
            t._observations = (
                compiled_obs if variant == "fused" else t.imagined_observations
            )
            progress(variant + "_warm_2048")
            warm = run(batch)
            if variant == "baseline":
                reference = warm
            else:
                result["full_batch_gradient_max_abs_error"] = compare(reference, warm)
                del reference
            del warm
            sync()
            times = []
            for j in range(repeats):
                start = time.perf_counter()
                run(batch)
                sync()
                times.append(time.perf_counter() - start)
            result[variant + "_seconds"] = times
            progress(variant + "_timed")
        assert counters == (t.actor_updates, t.model_updates)
        result["completed"] = True
        result["equivalence_passed"] = True
        progress("passed")
        return result
    finally:
        t._v24_step, t._observations = compiled_step, compiled_obs
        t.event_scheduler.rng = event_rng
