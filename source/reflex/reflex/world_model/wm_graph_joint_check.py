"""Stable-encoding WM eager/Graph + real Actor/Critic paired acceptance."""

import copy
import hashlib
from pathlib import Path

import torch

from .actor_execution_v32_check import rollout
from .diagnostic_watch_v35 import atomic_json, cold_watch
from .diagnostics_v31.full import compare, compile_counts, count_delta, distribution
from .diagnostics_v31.minimal import to_device
from .diagnostics_v31.state import Snapshot, digest, parameter_result
from .timing import timestamp


def check(t, output, repeats=5, warmups=2):
    if t.cfg.wm_encoding != "stable_heading_v1" or t._wm_rollout is None:
        raise ValueError("Require persistent stable encoding and H8 training fusion")
    if t.cfg.wm_compile_mode != "default":
        raise ValueError("Keep Actor dynamics at the verified default compile mode")
    if repeats < 1 or warmups < 1:
        raise ValueError("Positive repeat and warmup counts required")
    if not t.normalized:
        t.fit_normalizers()
    out = Path(output)
    path = out / "wm_graph_joint.json"
    if path.exists():
        raise FileExistsError(path)
    snapshot = Snapshot(t)
    model = to_device(copy.deepcopy(t.model.state_dict()), "cpu")
    optimizer = to_device(copy.deepcopy(t.model_opt.state_dict()), "cpu")
    amp = to_device(copy.deepcopy(t.prior.amp_state_dict()), "cpu")
    amp_previous = {
        key: getattr(t.prior, key).detach().cpu().clone()
        for key in ("_prev", "_prev_valid")
        if hasattr(t.prior, key)
    }
    gradients = [
        None if p.grad is None else p.grad.detach().cpu().clone()
        for p in t.model.parameters()
    ]
    counters = {
        k: getattr(t, k)
        for k in (
            "model_updates",
            "wm_rounds_completed",
            "wm_warmup_inherited",
            "_termination_probe_due",
        )
    }
    original = t._wm_rollout, t._wm_training_step
    result = {
        "completed": False,
        "config": copy.deepcopy(vars(t.cfg)),
        "scope": "Matched stable encoding; WM10+AMP+Actor/Critic, frozen Replay, restoration excluded",
        "diagnostic_actor_warmup_override": True,
        "formal_training": False,
        "variants": {k: {"warmup": [], "steady": []} for k in ("eager_wm", "graph_wm")},
    }

    def save():
        atomic_json(path, result)

    def reset(name):
        snapshot.reset()
        t.event_scheduler.static = copy.deepcopy(snapshot.static)
        t.model.load_state_dict(model)
        t.model_opt.load_state_dict(copy.deepcopy(optimizer))
        t.prior.load_amp_state_dict(copy.deepcopy(amp))
        for p, g in zip(t.model.parameters(), gradients):
            p.grad = None if g is None else g.to(p.device).clone()
        for key, value in counters.items():
            setattr(t, key, value)
        # Only this diagnostic bypasses the 500-round gate, restored on exit.
        t.wm_warmup_inherited = True
        t._wm_rollout, t._wm_training_step = (
            original if name == "graph_wm" else (None, None)
        )

    def cycle():
        if str(t.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(t.device)
        begin = timestamp(t.device)
        wm = t.fit_model()
        mid = timestamp(t.device)
        wm_peak = (
            torch.cuda.max_memory_allocated(t.device)
            if str(t.device).startswith("cuda")
            else 0
        )
        actor = t.update_actor()
        end = timestamp(t.device)
        return {
            "wm10_amp_seconds": mid - begin,
            "actor_critic_seconds": end - mid,
            "complete_seconds": end - begin,
            "wm": wm,
            "actor": actor,
            "peak_allocated_bytes": max(
                wm_peak, torch.cuda.max_memory_allocated(t.device)
            )
            if str(t.device).startswith("cuda")
            else 0,
        }

    def tensors():
        return {
            **parameter_result(t),
            **{
                "model/" + k: v.detach().cpu().clone()
                for k, v in t.model.state_dict().items()
            },
            **{
                f"model/grad/{i}": p.grad.detach().cpu().clone()
                for i, p in enumerate(t.model.parameters())
                if p.grad is not None
            },
        }

    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("wm_graph_joint_check.py", "trainer.py", "models.py", "physics.py")
    }
    save()
    try:
        for name, entry in result["variants"].items():
            for index in range(warmups):
                reset(name)

                def progress(value):
                    result["active_operation"] = value
                    save()

                with cold_watch(
                    t, f"warm/{name}/{index}", progress, out / "cold_stacks.log"
                ):
                    entry["warmup"].append(cycle())
                save()
        expected = None
        before = compile_counts()
        names = list(result["variants"])
        for j in range(repeats):
            for name in names[j % 2 :] + names[: j % 2]:
                reset(name)
                entry = result["variants"][name]
                entry["steady"].append(cycle())
                if j == 0:
                    actual = tensors()
                    if expected is None:
                        expected = actual
                    entry["actual_parameters_and_gradients"] = compare(expected, actual)
                save()
            print(f"[WM Graph joint] paired round {j + 1}/{repeats}", flush=True)
        result["steady_compile_delta"] = count_delta(compile_counts(), before)
        for entry in result["variants"].values():
            entry["seconds"] = {
                k: distribution([r[k] for r in entry["steady"]])
                for k in (
                    "wm10_amp_seconds",
                    "actor_critic_seconds",
                    "complete_seconds",
                )
            }
        expected_trajectory = None
        for name, entry in result["variants"].items():
            reset(name)
            t.fit_model()
            actual, entry["post_wm_actor_rollout"] = rollout(t)
            if expected_trajectory is None:
                expected_trajectory = actual
            entry["post_wm_trajectory_and_gradient"] = compare(
                expected_trajectory, actual
            )
            save()
        result["numeric_passed"] = all(
            e["actual_parameters_and_gradients"]["passed"]
            and e["post_wm_trajectory_and_gradient"]["passed"]
            for e in result["variants"].values()
        )
        result["performance_usable"] = not any(result["steady_compile_delta"].values())
        if str(t.device).startswith("cuda"):
            reset("graph_wm")
            # Warm the restored lifecycle before capture, without changing flags.
            cycle()
            reset("graph_wm")
            before = compile_counts()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                cycle()
            result["profile_compile_delta"] = count_delta(compile_counts(), before)
            result["cuda_graph_launch_events"] = sum(
                e.count
                for e in profiler.key_averages()
                if "cudagraphlaunch" in e.key.lower()
            )
            profiler.export_chrome_trace(str(out / "wm_graph_joint_trace.json"))
        result["completed"] = True
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        snapshot.restore()
        t.model.load_state_dict(model)
        t.model_opt.load_state_dict(optimizer)
        t.prior.load_amp_state_dict(amp)
        for key, value in amp_previous.items():
            getattr(t.prior, key).copy_(value.to(getattr(t.prior, key).device))
        for p, g in zip(t.model.parameters(), gradients):
            p.grad = None if g is None else g.to(p.device).clone()
        for key, value in counters.items():
            setattr(t, key, value)
        t._wm_rollout, t._wm_training_step = original
        result["model_restored"] = digest(t.model.state_dict()) == digest(model)
        result["actor_restored"] = digest(t.actor.state_dict()) == digest(
            snapshot.weights["actor"]
        )
        save()
    return result
