"""Paired H8 compile benchmark; restores model, Adam, gradients and RNG."""

import copy
import json
import statistics
from pathlib import Path

import torch

from .timing import timestamp


def check(t, output, repeats=3):
    if not t.cfg.analytic_soft_termination or not t.cfg.deterministic_event_apply:
        raise ValueError("Requires analytic deterministic WM")
    path = Path(output) / "wm_rollout_v30.json"
    if path.exists():
        raise FileExistsError(path)
    if not t.normalized:
        t.fit_normalizers()
    initial = copy.deepcopy(t.model.state_dict())
    optimizer = copy.deepcopy(t.model_opt.state_dict())
    gradients = [
        None if p.grad is None else p.grad.clone() for p in t.model.parameters()
    ]
    count, rounds = t.model_updates, t.wm_rounds_completed
    amp_state = copy.deepcopy(t.prior.amp_state_dict())
    amp_previous = {
        k: getattr(t.prior, k).clone()
        for k in ("_prev", "_prev_valid")
        if hasattr(t.prior, k)
    }
    old_sample, old_step, old_rollout = t.sample, t._wm_training_step, t._wm_rollout
    cpu_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if str(t.device).startswith("cuda") else None
    )
    result = {"completed": False, "variants": {}, "errors": {}}

    def restore():
        t.model.load_state_dict(copy.deepcopy(initial))
        t.model_opt.load_state_dict(copy.deepcopy(optimizer))
        t.model_updates, t.wm_rounds_completed = count, rounds
        t.prior.load_amp_state_dict(copy.deepcopy(amp_state))
        for name, value in amp_previous.items():
            getattr(t.prior, name).copy_(value)
        for p, grad in zip(t.model.parameters(), gradients):
            p.grad = None if grad is None else grad.clone()

    try:
        sequences = [old_sample(t.cfg.model_window) for _ in range(t.cfg.model_steps)]
        reference = None
        for label in ("baseline", "single_step", "h8_fused"):
            t._wm_training_step = (
                None
                if label == "baseline"
                else torch.compile(
                    t.v24_model_training_step,
                    backend=t.cfg.compile_backend,
                    fullgraph=True,
                    dynamic=False,
                )
            )
            t._wm_rollout = (
                torch.compile(
                    t.v30_model_rollout,
                    backend=t.cfg.compile_backend,
                    fullgraph=True,
                    dynamic=False,
                )
                if label == "h8_fused"
                else None
            )
            print(f"[V30 WM] start {label}", flush=True)
            rows = []
            for _ in range(repeats + 1):
                restore()
                index = [0]

                def sample(*args, _index=index, **kwargs):
                    row = sequences[_index[0]]
                    _index[0] += 1
                    return row

                t.sample = sample
                if cuda_rng is not None:
                    torch.cuda.reset_peak_memory_stats(t.device)
                begin = timestamp(t.device)
                metrics = t.fit_model()
                seconds = timestamp(t.device) - begin
                rows.append(
                    {
                        "seconds": seconds,
                        "metrics": metrics,
                        "peak_bytes": torch.cuda.max_memory_allocated(t.device)
                        if cuda_rng is not None
                        else None,
                    }
                )
            actual = {
                **{
                    f"parameter/{k}": v.detach().cpu().clone()
                    for k, v in t.model.state_dict().items()
                },
                **{
                    f"gradient/{i}": p.grad.detach().cpu().clone()
                    for i, p in enumerate(t.model.parameters())
                    if p.grad is not None
                },
            }
            if reference is None:
                reference = actual
            else:
                for key, value in actual.items():
                    result["errors"][f"{label}/{key}"] = float(
                        (value - reference[key]).abs().max()
                    )
                    try:
                        torch.testing.assert_close(value, reference[key])
                    except AssertionError as exc:
                        result.setdefault("failed_checks", []).append(f"{label}/{key}")
                        result.setdefault("failure_details", {})[f"{label}/{key}"] = (
                            str(exc)
                        )
            result["variants"][label] = {
                "cold": rows[0],
                "steady": rows[1:],
                "median_seconds": statistics.median(r["seconds"] for r in rows[1:]),
            }
            result["variants"][label]["numeric_passed"] = not any(
                k.startswith(label + "/") for k in result.get("failed_checks", [])
            )
            print(
                f"[V30 WM] done {label}: {result['variants'][label]['median_seconds']:.6f}s",
                flush=True,
            )
            path.write_text(json.dumps(result, indent=2))
        result["completed"] = True
        result["numeric_passed"] = not result.get("failed_checks")
    finally:
        restore()
        t.sample, t._wm_training_step, t._wm_rollout = old_sample, old_step, old_rollout
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        path.write_text(json.dumps(result, indent=2))
    return result
