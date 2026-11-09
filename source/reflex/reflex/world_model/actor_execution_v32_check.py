"""Paired frozen-Replay V32 checks, no checkpoint writes or formal training."""

import copy
import hashlib
from contextlib import nullcontext
from pathlib import Path

import torch

from . import joint
from .diagnostic_watch_v35 import atomic_json
from .diagnostics_v31.full import compare, compile_counts, count_delta, distribution
from .diagnostics_v31.state import Snapshot, digest, parameter_result
from .numerical_risk_v35 import actor_risk, trajectory_risk
from .timing import timestamp
from .trainer import freeze

FLAGS = ("compact_actor_event_inputs", "packed_pose_carry")
VARIANTS = {
    "reference": (False, False),
    "event_gathers": (True, False),
    "pose_carry": (False, True),
    "combined": (True, True),
}


def rollout(t):
    """Real H32/critic bootstrap/loss; snapshots and hashing are outside timing."""
    batch = t.sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
    with freeze(t.model, t.termination, t.prior._disc, t.critic):
        begin = timestamp(t.device)
        imagined = t.imagine(batch, trace=True)
        values = imagined.get("actor_values")
        if values is None and t.cfg.batch_imagination_critic:
            obs = imagined.get("critic_obs", imagined["obs"])
            from .critic_values import evaluate

            values = evaluate(
                t.critic, obs, t.cfg.critic_value_chunk, t.cfg.checkpoint_actor_critic
            )
        objective = -joint.returns(imagined, t.critic, t.cfg.gamma, values).mean()
        if t.cfg.actor_horizon_scale:
            objective = objective / t.cfg.horizon
        middle = timestamp(t.device)
        gradients = torch.autograd.grad(objective, tuple(t.actor.parameters()))
        end = timestamp(t.device)
    data = {k: v.detach().cpu().clone() for k, v in imagined.items()}
    data["actor_gradient"] = torch.cat([g.detach().flatten().cpu() for g in gradients])
    modes = imagined.get("predicted_launch_probability")
    before = torch.cat((batch["launched"][None].float(), modes[:-1]))
    coverage = {
        "waiting_to_launch": int(((modes > 0.5) & (before <= 0.5)).sum()),
        "return_to_waiting": int(((modes <= 0.5) & (before > 0.5)).sum()),
        "truncations": int(imagined["truncated"].sum()),
    }
    return data, {
        "forward_seconds": middle - begin,
        "BPTT_seconds": end - middle,
        "sample_sha256": digest(batch),
        "gradient_norm": float(data["actor_gradient"].norm()),
        "coverage": coverage,
        "note": "Single traced semantic rollout, not steady performance estimate",
    }


def check(
    t,
    output,
    repeats=20,
    warmups=2,
    *,
    variants=None,
    flag_names=None,
    apply_variant=None,
    cold_context=None,
    report_name="actor_execution_v32.json",
):
    variants = VARIANTS if variants is None else variants
    flag_names = FLAGS if flag_names is None else flag_names
    if not variants or next(iter(variants)) != "reference":
        raise ValueError("First variant must be the common reference")

    def context(label, result):
        return nullcontext() if cold_context is None else cold_context(label, result)

    if repeats < 1 or warmups < 1:
        raise ValueError("Positive repeats and warmups required")
    if not (
        t.cfg.condition_external_events
        and t.cfg.deterministic_event_apply
        and t.cfg.minimal_event_view
        and t.cfg.batch_imagination_critic
        and t.model.state_only
    ):
        raise ValueError("Use the verified native state-WM/packed-critic configuration")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / report_name
    if path.exists():
        raise FileExistsError(path)
    snapshot = Snapshot(t)
    original_flags = tuple(getattr(t.cfg, name) for name in flag_names)
    model_before = digest(t.model.state_dict())
    actor_before = digest(t.actor.state_dict())
    result = {
        "completed": False,
        "config": copy.deepcopy(vars(t.cfg)),
        "device": str(t.device),
        "torch": str(torch.__version__),
        "checkpoint_saved": False,
        "timing": "Interleaved full real update; snapshot restoration excluded; cold warmups separate",
        "variants": {name: {"warmup": [], "steady": []} for name in variants},
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "trainer.py",
                "actor_execution_v32.py",
                "actor_execution_v32_check.py",
            )
        },
    }
    if str(t.device).startswith("cuda"):
        result["gpu"] = torch.cuda.get_device_name(t.device)
    sample_present, sample_saved = "sample" in t.__dict__, t.__dict__.get("sample")
    sampler, observed = t.sample, {}

    def sample(*args, **kwargs):
        batch = sampler(*args, **kwargs)
        if kwargs.get("starts", False):
            observed["starts"] = batch
        return batch

    def reset(name):
        snapshot.reset()
        # Native callbacks may cache posture ids. Do not inherit trial mutations.
        t.event_scheduler.static = copy.deepcopy(snapshot.static)
        for key, value in zip(flag_names, variants[name]):
            setattr(t.cfg, key, value)
        if apply_variant is not None:
            apply_variant(name)

    atomic_json(path, result)
    t.sample = sample
    reference_update = None
    try:
        # Warm each stride/layout outside timing comparisons; every update discarded.
        for name, entry in result["variants"].items():
            for _ in range(warmups):
                reset(name)
                begin = timestamp(t.device)
                with context(f"warm/{name}", result):
                    t.update_actor()
                entry["warmup"].append(timestamp(t.device) - begin)
                observed.clear()
        reset("reference")
        with context("trajectory/reference", result):
            expected, details = rollout(t)
        result["variants"]["reference"]["rollout"] = details
        result["sample_sha256"] = details["sample_sha256"]
        for name, entry in result["variants"].items():
            reset(name)
            with context(f"trajectory/{name}", result):
                actual, entry["rollout"] = rollout(t)
            entry["trajectory_and_raw_gradient"] = compare(expected, actual)
            entry["numerical_risk"] = trajectory_risk(expected, actual)
            atomic_json(path, result)
            del actual
            observed.clear()
        del expected
        names = list(variants)
        before = compile_counts()
        for j in range(repeats):
            # Rotate order to avoid systematically assigning boost/thermal drift.
            for name in names[j % len(names) :] + names[: j % len(names)]:
                entry = result["variants"][name]
                reset(name)
                if str(t.device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(t.device)
                begin = timestamp(t.device)
                metrics = t.update_actor()
                elapsed = timestamp(t.device) - begin
                row = {"complete_seconds": elapsed, **metrics}
                if str(t.device).startswith("cuda"):
                    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                        t.device
                    )
                    row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                        t.device
                    )
                entry["steady"].append(row)
                if j == 0:
                    actual_update = parameter_result(t)
                    if reference_update is None:
                        reference_update = actual_update
                    entry["actual_parameter_and_gradient"] = compare(
                        reference_update, actual_update
                    )
                    names_and_params = tuple(t.actor.named_parameters())

                    def convert_update(value, names_and_params=names_and_params):
                        return {
                            **{
                                f"parameter/{key}": value[f"actor/{key}"]
                                for key, _ in names_and_params
                            },
                            **{
                                f"gradient/{i}": value[f"actor/grad/{i}"]
                                for i in range(len(names_and_params))
                            },
                        }

                    entry["update_risk"] = actor_risk(
                        convert_update(reference_update),
                        convert_update(actual_update),
                        snapshot.weights["actor"],
                        names_and_params,
                    )
                    entry["sample_sha256"] = digest(observed["starts"])
                    if actual_update is not reference_update:
                        del actual_update
                observed.clear()
            atomic_json(path, result)
            print(f"[V32] paired round {j + 1}/{repeats}", flush=True)
        result["steady_compile_delta"] = count_delta(compile_counts(), before)
        for entry in result["variants"].values():
            entry["wall_seconds"] = distribution(
                [r["complete_seconds"] for r in entry["steady"]]
            )
            entry["numeric_passed"] = (
                entry["trajectory_and_raw_gradient"]["passed"]
                and entry["actual_parameter_and_gradient"]["passed"]
                and entry["sample_sha256"] == result["sample_sha256"]
                and entry["rollout"]["sample_sha256"] == result["sample_sha256"]
            )
        result["completed"] = True
        result["numeric_passed"] = all(
            v["numeric_passed"] for v in result["variants"].values()
        )
        result["performance_usable"] = not any(result["steady_compile_delta"].values())
        result["limitations"] = [
            "CPU results do not validate CUDA/Inductor or speed gains.",
            "Coverage counts do not guarantee all native contact/rethrow subcases.",
            "No configuration is promoted automatically, even if numeric checks pass.",
        ]
    finally:
        if sample_present:
            t.sample = sample_saved
        else:
            del t.sample
        observed.clear()
        for key, value in zip(flag_names, original_flags):
            setattr(t.cfg, key, value)
        snapshot.restore()
        result["model_unchanged"] = digest(t.model.state_dict()) == model_before
        result["actor_restored"] = digest(t.actor.state_dict()) == actor_before
        if result.get("completed"):
            result["numeric_passed"] &= (
                result["model_unchanged"] and result["actor_restored"]
            )
        atomic_json(path, result)
    return result
