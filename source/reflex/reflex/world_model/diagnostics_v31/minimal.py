"""Portable H32 Actor+Physical-WM compute control, NOT task training.

The same step, input tensors, objective and initialized Adam state are used by
all drivers. Only the compilation boundary changes; H32 remains sequential.
"""

import argparse
import copy
import json
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..compact_context import ContextSpec
from ..models import (
    CompactContextWorldModel,
    DeterministicEventWorldModel,
    Standardizer,
    mlp,
)
from ..physics import ActionMap
from .full import compare, distribution
from .state import digest


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(to_device(v, device) for v in value)
    return value


def flatten_tensors(value, prefix="input"):
    if torch.is_tensor(value):
        return {prefix: value}
    result = {}
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in items:
        result.update(flatten_tensors(child, f"{prefix}/{key}"))
    return result


def export(t, output):
    """Only tensors/primitive metadata; no pickle of simulator or callbacks."""
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    if not (t.cfg.condition_external_events and t.cfg.deterministic_event_apply):
        raise ValueError(
            "Minimal capsule currently supports deterministic conditioned state WM"
        )
    if t.cfg.wm_mlp_precision != "fp32":
        raise ValueError(
            "Export the FP32 baseline; precision changes are a separate experiment"
        )
    rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if str(t.device).startswith("cuda") else None
    )
    old_cache = dict(t.replay._index_cache)
    old_seconds = t.replay_seconds
    try:
        with torch.no_grad():
            batch = t.sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
            inputs = {
                k: batch[k].detach().cpu().clone()
                for k in (
                    "state",
                    "previous_action",
                    "context",
                    "blend",
                    "launched",
                    "scale",
                    "offset",
                    "down",
                    "up",
                    "limits",
                )
            }
            inputs["obs0"] = t.observation(batch).detach().cpu()
            inputs["pre"] = (
                t.event_scheduler.codec.pre_features(
                    batch["event_state"], batch["episode_steps"], t.task.dt
                )
                .detach()
                .cpu()
            )
        contract = {
            k: t.contract[k]
            for k in ("context_dim", "com_indices", "context_schema", "body_names")
            if k in t.contract
        }
        capsule = {
            "version": 1,
            "kind": "reflex",
            "config": copy.deepcopy(vars(t.cfg)),
            "contract": contract,
            "dt": t.model.dt,
            "action_map": {
                "arm_columns": t.action_map.arms.detach().cpu().tolist(),
                "ramp": t.action_map.ramp,
                "delta_min": t.action_map.delta_min,
                "delta_gate": t.action_map.delta_gate,
                "base_blend": t.action_map.base_blend,
            },
            "actor": to_device(t.actor.state_dict(), "cpu"),
            "model": to_device(t.model.state_dict(), "cpu"),
            "obs_norm": to_device(t.obs_norm.state_dict(), "cpu"),
            "inputs": inputs,
            "input_sha256": digest(inputs),
            "full_start_sha256": digest(batch),
            "source_actor_sha256": digest(t.actor.state_dict()),
            "source_model_sha256": digest(t.model.state_dict()),
            "note": "Fixed differentiable observation proxy, held event features, Physical transitions only; not original task semantics",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(capsule, path)
        return {
            k: capsule[k]
            for k in (
                "version",
                "kind",
                "input_sha256",
                "full_start_sha256",
                "source_actor_sha256",
                "source_model_sha256",
                "note",
            )
        }
    finally:
        t.replay._index_cache.clear()
        t.replay._index_cache.update(old_cache)
        t.replay_seconds = old_seconds
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


class ReflexStep(nn.Module):
    def __init__(self, capsule, device):
        super().__init__()
        cfg = capsule["config"]
        self.actor = mlp(179, 29, cfg["hidden"], layers=cfg["actor_hidden_dims"]).to(
            device
        )
        kwargs = {
            "hidden_layers": cfg["wm_hidden_dims"] or [2048] * 3,
            "com_indices": capsule["contract"].get("com_indices"),
            "dt": capsule["dt"],
        }
        self.model = (
            CompactContextWorldModel(
                ContextSpec.from_contract(capsule["contract"]), **kwargs
            )
            if cfg["wm_context_mode"] != "full"
            else DeterministicEventWorldModel(
                capsule["contract"]["context_dim"], **kwargs
            )
        ).to(device)
        self.obs_norm = Standardizer(179).to(device)
        self.action_map = ActionMap(**capsule["action_map"]).to(device)
        self.actor.load_state_dict(capsule["actor"])
        self.model.load_state_dict(capsule["model"])
        self.obs_norm.load_state_dict(capsule["obs_norm"])
        self.model.requires_grad_(False)
        self.data = to_device(capsule["inputs"], device)
        self.wm_call = None

    def initial(self):
        return self.data["state"], self.data["previous_action"], self.data["blend"]

    def state_vector(self, state):
        return state

    def forward(self, state, previous, blend):
        d = self.data
        # Shared diagnostic projection, deliberately independent of real FK.
        obs = (
            d["obs0"]
            + 0.01 * F.pad(state - d["state"], (0, 95))
            + 0.01 * F.pad(previous - d["previous_action"], (150, 0))
        )
        raw = self.actor(self.obs_norm(obs))
        target, blend = self.action_map(
            raw,
            blend,
            d["launched"],
            d["scale"],
            d["offset"],
            d["down"],
            d["up"],
            d["limits"],
        )
        call = self.model.predict_physical if self.wm_call is None else self.wm_call
        return call(state, target, d["context"], d["pre"]), raw, blend


def rollout(step, call, horizon):
    state, previous, aux = step.initial()
    base = step.state_vector(state)
    vectors, actions = [], []
    first = None
    for _ in range(horizon):
        state, previous, aux = call(state, previous, aux)
        if first is None:
            first = previous
        vectors.append(step.state_vector(state))
        actions.append(previous)
    states, actions = torch.stack(vectors), torch.stack(actions)
    state_loss = (states - base).square().mean()
    # Nonzero action target prevents a zero-output policy from hiding backward.
    loss = state_loss + 0.001 * (actions - 0.1).square().mean()
    return loss, state_loss, states, actions, first


def common_adam(actor):
    opt = torch.optim.Adam(actor.parameters(), lr=2e-4, betas=(0.9, 0.95))
    # Allocate identical zero moments outside timed intervals (also for native).
    for p in actor.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    opt.zero_grad(set_to_none=True)
    return opt


def check(
    factory,
    output,
    device="cpu",
    horizon=32,
    repeats=20,
    warmups=2,
    backend="inductor",
    drivers=("eager", "segmented", "single_step"),
    metadata=None,
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "minimal_actor_v31.json"
    if path.exists():
        raise FileExistsError(path)
    if repeats < 1 or warmups < 1 or horizon < 1:
        raise ValueError("Positive counts required")
    original_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None
    )
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    result = {
        "completed": False,
        "kind": "standalone_diagnostic_not_task_training",
        "metadata": metadata or {},
        "horizon": horizon,
        "precision": "fp32_tf32_off",
        "drivers": {},
        "torch": torch.__version__,
        "device": str(device),
        "tolerance": "torch.testing.assert_close dtype defaults; no relaxed tolerances",
    }
    reference = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for driver in drivers:
            if driver not in ("eager", "segmented", "single_step"):
                raise ValueError("Unknown driver")
            torch.set_rng_state(original_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            step = factory()
            if any(p.dtype != torch.float32 for p in step.parameters()):
                raise ValueError("Minimal comparison requires FP32 parameters")
            initial_signature = {
                "weights": digest(step.state_dict()),
                "inputs": digest(flatten_tensors(step.data)),
            }
            if result.get("initial_signature", initial_signature) != initial_signature:
                raise ValueError(
                    "Drivers did not receive identical model weights and input tensors"
                )
            result["initial_signature"] = initial_signature
            model_initial = copy.deepcopy(step.model.state_dict())
            actor_initial = copy.deepcopy(step.actor.state_dict())
            opt = common_adam(step.actor)
            opt_initial = copy.deepcopy(opt.state_dict())
            iteration_cpu_rng = torch.get_rng_state()
            iteration_cuda_rng = (
                torch.cuda.get_rng_state_all() if cuda_rng is not None else None
            )
            step.wm_call = (
                torch.compile(
                    step.model.predict_physical
                    if hasattr(step.model, "predict_physical")
                    else step.model.step,
                    fullgraph=True,
                    dynamic=False,
                    backend=backend,
                )
                if driver == "segmented"
                else None
            )
            call = (
                torch.compile(step, fullgraph=True, dynamic=False, backend=backend)
                if driver == "single_step"
                else step
            )
            rows = []
            for iteration in range(warmups + repeats):
                step.actor.load_state_dict(actor_initial)
                step.model.load_state_dict(model_initial)
                opt.load_state_dict(copy.deepcopy(opt_initial))
                opt.zero_grad(set_to_none=True)
                torch.set_rng_state(iteration_cpu_rng)
                if iteration_cuda_rng is not None:
                    torch.cuda.set_rng_state_all(iteration_cuda_rng)
                if str(device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(device)
                begin = clock(device)
                loss, _, _, _, _ = rollout(step, call, horizon)
                middle = clock(device)
                loss.backward()
                back = clock(device)
                norm = nn.utils.clip_grad_norm_(
                    step.actor.parameters(), 1.0, error_if_nonfinite=True
                )
                nn.utils.clip_grad_value_(step.actor.parameters(), 0.1)
                opt.step()
                end = clock(device)
                rows.append(
                    {
                        "forward_seconds": middle - begin,
                        "backward_seconds": back - middle,
                        "clip_adam_seconds": end - back,
                        "complete_seconds": end - begin,
                        "gradient_norm": float(norm),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                        if str(device).startswith("cuda")
                        else None,
                    }
                )
            actual = {
                f"parameter/{k}": v.detach().cpu().clone()
                for k, v in step.actor.state_dict().items()
            }
            actual.update(
                {
                    f"gradient/{i}": p.grad.detach().cpu().clone()
                    for i, p in enumerate(step.actor.parameters())
                }
            )
            step.actor.load_state_dict(actor_initial)
            step.model.load_state_dict(model_initial)
            torch.set_rng_state(iteration_cpu_rng)
            if iteration_cuda_rng is not None:
                torch.cuda.set_rng_state_all(iteration_cuda_rng)
            loss, state_loss, states, actions, first = rollout(step, call, horizon)
            actual.update(
                states=states.detach().cpu(),
                actions=actions.detach().cpu(),
                loss=loss.detach().cpu(),
            )
            terminal = (
                (states[-1] - step.state_vector(step.initial()[0])).square().mean()
            )
            physical_vjp = torch.autograd.grad(terminal, first, allow_unused=True)[0]
            vjp_norm = float(physical_vjp.norm()) if physical_vjp is not None else 0.0
            if reference is None:
                reference = actual
            numeric = compare(reference, actual)
            result["drivers"][driver] = {
                "numeric": numeric,
                "warmup": rows[:warmups],
                "steady": rows[warmups:],
                "wall_seconds": distribution(
                    [r["complete_seconds"] for r in rows[warmups:]]
                ),
                "state_loss_to_first_action_vjp_norm": vjp_norm,
                "physical_bptt_nonzero": vjp_norm > 0,
                "actor_parameter_count": sum(
                    p.numel() for p in step.actor.parameters()
                ),
                "wm_parameter_count": sum(p.numel() for p in step.model.parameters()),
                "wm_frozen": all(not p.requires_grad for p in step.model.parameters()),
                "batch": len(actions[0]),
                "state_scalar_count": states.shape[-1],
            }
            print(
                f"[V31 minimal] {driver}: {statistics.median(r['complete_seconds'] for r in rows[warmups:]):.6f}s; numeric={numeric['passed']}",
                flush=True,
            )
            path.write_text(json.dumps(result, indent=2))
            del (
                actual,
                loss,
                state_loss,
                states,
                actions,
                first,
                physical_vjp,
                call,
                step,
                opt,
            )
        result["completed"] = True
        result["numeric_passed"] = all(
            v["numeric"]["passed"] for v in result["drivers"].values()
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        torch.set_rng_state(original_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        path.write_text(json.dumps(result, indent=2))
    return result


def clock(device):
    # Only diagnostic phase boundaries synchronize, never individual modules.
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capsule")
    p.add_argument("--tracker_repo")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--backend", default="inductor")
    p.add_argument(
        "--drivers", nargs="+", default=["eager", "segmented", "single_step"]
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if bool(a.capsule) == bool(a.tracker_repo):
        p.error("Choose one capsule or tracker_repo")
    if a.capsule:
        capsule = torch.load(a.capsule, map_location="cpu", weights_only=True)
        if capsule.get("version") != 1 or capsule.get("kind") != "reflex":
            raise ValueError("Unsupported capsule schema")
        if len(capsule["inputs"]["state"]) != a.batch:
            raise ValueError("Requested batch must match exported capsule")
        factory = lambda: ReflexStep(capsule, a.device)
        metadata = {
            k: capsule[k]
            for k in (
                "kind",
                "input_sha256",
                "full_start_sha256",
                "source_actor_sha256",
                "source_model_sha256",
                "note",
            )
        }
    else:
        from .tracker_native import factory_from_repo

        factory, metadata = factory_from_repo(a.tracker_repo, a.device, a.batch)
    check(
        factory,
        a.output,
        a.device,
        a.horizon,
        a.repeats,
        a.warmups,
        a.backend,
        tuple(a.drivers),
        metadata,
    )


if __name__ == "__main__":
    main()
