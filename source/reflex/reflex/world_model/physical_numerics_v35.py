"""Same-input stage/VJP diagnosis plus controlled H32/Adam boundary trials."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .diagnostics_v31.full import compare
from .diagnostics_v31.minimal import ReflexStep
from .diagnostics_v31.minimal import check as minimal_check
from .diagnostics_v31.state import digest
from .physical_compile_v35 import components


def compare_stage(eager, compiled, inputs):
    """Separate local value/gradient errors from propagated input drift."""
    values, gradients = [], []
    for call in (eager, compiled):
        args = tuple(v.detach().clone().requires_grad_(True) for v in inputs)
        output = call(*args)
        # Fixed, nonuniform cotangent; no RNG and no optimizer state changes.
        cotangent = torch.linspace(
            0.5, 1.5, output.shape[-1], device=output.device, dtype=output.dtype
        )
        loss = (output * cotangent).sum() / output.numel()
        grads = torch.autograd.grad(loss, args, allow_unused=True)
        values.append({"output": output.detach().cpu().clone()})
        gradients.append(
            {
                str(i): (torch.zeros_like(a) if g is None else g).detach().cpu().clone()
                for i, (a, g) in enumerate(zip(args, grads))
            }
        )
    return {"value": compare(*values), "input_vjp": compare(*gradients)}


def stage_probe(
    step, steps=(1, 6, 16, 32), backend="inductor", progress=None, mlp_layers=False
):
    if not steps or min(steps) < 1:
        raise ValueError("Positive probe steps required")
    c = components(step.model)
    if mlp_layers:
        c.update(
            {
                f"layer/{i}/{type(layer).__name__}": layer
                for i, layer in enumerate(step.model.net)
            }
        )
    compiled = {
        name: torch.compile(fn, fullgraph=True, dynamic=False, backend=backend)
        for name, fn in c.items()
    }
    state, previous, blend = step.initial()
    d = step.data
    rows = []
    for index in range(1, max(steps) + 1):
        # Exactly the eager diagnostic Actor projection/control used in H32.
        obs = (
            d["obs0"]
            + 0.01 * torch.nn.functional.pad(state - d["state"], (0, 95))
            + 0.01 * torch.nn.functional.pad(previous - d["previous_action"], (150, 0))
        )
        raw = step.actor(step.obs_norm(obs))
        target, next_blend = step.action_map(
            raw,
            blend,
            d["launched"],
            d["scale"],
            d["offset"],
            d["down"],
            d["up"],
            d["limits"],
        )
        with torch.no_grad():
            features = c["features"](state, target, d["context"], d["pre"])
            encoded = step.model.inputs(features)
            delta = step.model.net(encoded)
        if index in steps:
            args = {
                "features": (state, target, d["context"], d["pre"]),
                "normalization": (features,),
                "mlp": (encoded,),
                "integrate": (state, delta, d["context"]),
                "encoder": (state, target, d["context"], d["pre"]),
            }
            if mlp_layers:
                value = encoded
                with torch.no_grad():
                    for i, layer in enumerate(step.model.net):
                        args[f"layer/{i}/{type(layer).__name__}"] = (value,)
                        value = layer(value)
            row = {"step": index, "components": {}}
            rows.append(row)
            for name, fn in c.items():
                row["components"][name] = compare_stage(fn, compiled[name], args[name])
                if progress is not None:
                    progress({"completed": False, "rows": rows})
        # Teacher-forced reference trajectory, not the compiled prediction.
        with torch.no_grad():
            state = step.model.predict_physical(state, target, d["context"], d["pre"])
        previous, blend = raw.detach(), next_blend.detach()
    return {
        "completed": True,
        "steps": list(steps),
        "rows": rows,
        "note": "Isolated components at identical eager inputs; not timings or proof of a whole fused graph. All state/context derivatives are probed; H32/actual Adam checked separately.",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capsule", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", default="inductor")
    p.add_argument("--steps", nargs="+", type=int, default=[1, 6, 16, 32])
    p.add_argument(
        "--drivers", nargs="+", default=["eager", "segmented", "wm_mlp", "wm_layerwise"]
    )
    p.add_argument("--components-only", action="store_true")
    p.add_argument("--mlp-layers", action="store_true")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmups", type=int, default=2)
    a = p.parse_args()
    if not a.drivers or a.drivers[0] != "eager":
        p.error("The first driver must be eager, the common reference")
    if a.repeats < 1 or a.warmups < 1 or not a.steps or min(a.steps) < 1:
        p.error("Positive steps/repeats/warmups required")
    from .physical_compile_v35 import DRIVERS

    if set(a.drivers) - {"eager", "segmented", "single_step", *DRIVERS}:
        p.error("Unknown compilation boundary")
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    capsule = torch.load(a.capsule, weights_only=True, map_location="cpu")
    if capsule.get("kind") != "reflex" or capsule.get("version") != 1:
        raise ValueError("Require original Reflex V31 capsule")
    out.mkdir(parents=True)
    metadata = {
        "capsule_sha256": hashlib.sha256(Path(a.capsule).read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "backend": a.backend,
        "fp_fusion_env": os.environ.get("TRITON_DEFAULT_FP_FUSION", "unset"),
        "training_changes": False,
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "physical_compile_v35.py",
                "physical_numerics_v35.py",
                "models.py",
                "physics.py",
                "diagnostics_v31/minimal.py",
            )
        },
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2))
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else None
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        step = ReflexStep(capsule, a.device)
        initial = digest(step.state_dict())

        def progress(partial):
            (out / "physical_components.partial.json").write_text(
                json.dumps(partial, indent=2)
            )
            last = partial["rows"][-1]
            print(
                f"[V35 components] step={last['step']} stage={list(last['components'])[-1]}",
                flush=True,
            )

        report = stage_probe(step, a.steps, a.backend, progress, a.mlp_layers)
        report["model_unchanged"] = digest(step.state_dict()) == initial
        (out / "physical_components.json").write_text(json.dumps(report, indent=2))
        del step
        if a.components_only:
            print(
                json.dumps(
                    {
                        "completed": True,
                        "kind": "components_only_not_training_validation",
                        "output": str(out),
                    }
                ),
                flush=True,
            )
            return
        result = minimal_check(
            lambda: ReflexStep(capsule, a.device),
            out / "h32",
            device=a.device,
            horizon=32,
            repeats=a.repeats,
            warmups=a.warmups,
            backend=a.backend,
            drivers=a.drivers,
            metadata=metadata,
        )
        print(
            json.dumps(
                {
                    "completed": result["completed"],
                    "numeric_passed": result["numeric_passed"],
                    "output": str(out),
                }
            ),
            flush=True,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


if __name__ == "__main__":
    main()
