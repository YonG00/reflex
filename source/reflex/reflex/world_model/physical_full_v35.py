"""Real-Replay H8 and complete-task H32 numerical gates, no checkpoint writes."""

import copy
from pathlib import Path

import torch

from .actor_execution_v32_check import rollout
from .diagnostic_watch_v35 import atomic_json, cold_watch
from .diagnostics_v31.full import compare, distribution
from .diagnostics_v31.state import Snapshot, digest, parameter_result
from .physical_compile_v35 import build
from .timing import timestamp


def check(t, output, repeats=3, warmups=1, stage="all", trace_seconds=0):
    if stage not in ("all", "wm", "actor"):
        raise ValueError("Unknown verification stage")
    if not t.cfg.analytic_soft_termination or t.cfg.fuse_actor_core:
        raise ValueError("Use the analytic packed-head baseline, not Actor core fusion")
    if repeats < 1 or warmups < 1:
        raise ValueError("Positive repetitions required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "physical_full_v35.json"
    if path.exists():
        raise FileExistsError(path)
    snapshot = Snapshot(t)
    model_state = copy.deepcopy(t.model.state_dict())
    model_opt = copy.deepcopy(t.model_opt.state_dict())
    gradients = [
        None if p.grad is None else p.grad.clone() for p in t.model.parameters()
    ]
    predict_present = "predict_physical" in t.model.__dict__
    original_predict = t.model.predict_physical
    original_dynamics = t._dynamics
    original_static = copy.deepcopy(t.event_scheduler.static)
    initial_model_hash = digest(model_state)
    candidate = build(t.model, "wm_native_split", t.cfg.compile_backend)
    variants = {
        "eager_physical": original_predict,
        "configured_baseline": original_dynamics,
        "native_split": candidate,
    }
    result = {
        "completed": False,
        "requested_stage": stage,
        "model_restored": False,
        "checkpoint_saved": False,
        "config": copy.deepcopy(vars(t.cfg)),
        "variants": {name: {} for name in variants},
        "scope": "actual Replay H8 WM and complete Actor/AMP/critic/reward/events",
        "timing_note": "Short frozen-update control; not end-to-end training throughput",
    }

    def save():
        atomic_json(path, result)

    def progress(value):
        result["active_operation"] = value
        save()

    def watching(label):
        result["current_stage"] = label
        save()
        return cold_watch(t, label, progress, out / "cold_stacks.log", trace_seconds)

    save()

    def reset(call):
        snapshot.reset()
        t.event_scheduler.static = copy.deepcopy(original_static)
        t.model.load_state_dict(model_state)
        t.model_opt.load_state_dict(copy.deepcopy(model_opt))
        for p, gradient in zip(t.model.parameters(), gradients):
            p.grad = None if gradient is None else gradient.clone()
        t.model.predict_physical = call
        t._dynamics = call

    def wm_step(seq):
        t.model_opt.zero_grad(set_to_none=True)
        post, physical, termination, post_loss = t.v30_model_rollout(seq)
        losses = physical + t.cfg.term_loss_weight * termination
        mask = seq.get("loss_mask")
        loss = (
            losses.mean()
            if mask is None
            else (torch.where(mask, losses, 0).sum(0) / mask.sum(0).clamp_min(1)).mean()
        )
        loss.backward()
        values = {
            "post_state": post.detach().cpu().clone(),
            "physical_loss": physical.detach().cpu().clone(),
            "post_loss": post_loss.detach().cpu().clone(),
            "loss": loss.detach().cpu().clone(),
        }
        for i, p in enumerate(t.model.parameters()):
            if p.grad is not None:
                values[f"raw_gradient/{i}"] = p.grad.detach().cpu().clone()
        torch.nn.utils.clip_grad_norm_(
            t.model.parameters(), t.cfg.model_grad_clip, error_if_nonfinite=True
        )
        t.model_opt.step()
        values.update(
            {
                f"parameter/{k}": v.detach().cpu().clone()
                for k, v in t.model.state_dict().items()
            }
        )
        return values

    try:
        if stage != "actor":
            reset(original_predict)
            seq = t.sample(t.cfg.model_window)
            result["wm_sequence_sha256"] = digest(seq)
            expected_wm = None
            for name, call in variants.items():
                reset(call)
                with watching(f"wm/{name}"):
                    actual = wm_step(seq)
                if expected_wm is None:
                    expected_wm = actual
                result["variants"][name].update(
                    {"wm_h8_gradient_and_adam": compare(expected_wm, actual)}
                )
                save()
                print(f"[V35 full] {name} WM done", flush=True)
            del expected_wm
            if stage == "wm":
                result["completed"] = True
                result["candidate_wm_numeric_passed"] = result["variants"][
                    "native_split"
                ]["wm_h8_gradient_and_adam"]["passed"]
                save()
                return result
        for name, call in variants.items():
            for _ in range(warmups):
                reset(call)
                with watching(f"warm/{name}"):
                    t.update_actor()
            save()
            print(f"[V35 full] {name} warm", flush=True)
        expected_rollout = expected_update = None
        for name, call in variants.items():
            reset(call)
            with watching(f"trajectory/{name}"):
                actual, details = rollout(t)
            if expected_rollout is None:
                expected_rollout = actual
            result["variants"][name].update(
                trajectory_and_raw_actor_gradient=compare(expected_rollout, actual),
                rollout=details,
                steady=[],
            )
            del actual
            save()
        del expected_rollout
        for _ in range(repeats):
            for name, call in variants.items():
                reset(call)
                begin = timestamp(t.device)
                metrics = t.update_actor()
                elapsed = timestamp(t.device) - begin
                entry = result["variants"][name]
                entry["steady"].append({"seconds": elapsed, **metrics})
                actual = parameter_result(t)
                if expected_update is None:
                    expected_update = actual
                entry["actual_actor_critic_update"] = compare(expected_update, actual)
                print(f"[V35 full] {name} actual update", flush=True)
            save()
        for entry in result["variants"].values():
            keys = ["trajectory_and_raw_actor_gradient", "actual_actor_critic_update"]
            if stage == "all":
                keys.append("wm_h8_gradient_and_adam")
            entry["numeric_passed"] = all(entry[key]["passed"] for key in keys)
            entry["wall_seconds"] = distribution(
                [row["seconds"] for row in entry["steady"]]
            )
        result["completed"] = True
        result[
            "candidate_numeric_passed"
            if stage == "all"
            else "candidate_actor_numeric_passed"
        ] = result["variants"]["native_split"]["numeric_passed"]
    except BaseException as exc:
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise
    finally:
        t.model.load_state_dict(model_state)
        t.model_opt.load_state_dict(model_opt)
        for p, gradient in zip(t.model.parameters(), gradients):
            p.grad = None if gradient is None else gradient.clone()
        if predict_present:
            t.model.predict_physical = original_predict
        else:
            del t.model.predict_physical
        t._dynamics = original_dynamics
        snapshot.restore()
        result["model_restored"] = digest(t.model.state_dict()) == initial_model_hash
        save()
    return result
