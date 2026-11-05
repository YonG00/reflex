"""Compare pure dynamics/FK eager vs compiled outputs and gradients on a saved replay.

Run after enabling GPU. This does not update any policy/model or mutate the checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from reflex.world_model.models import StateWorldModel
from reflex.world_model.physics import Kinematics
from reflex.world_model.replay import Replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--backend", default="inductor")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    c, conf = ck["contract"], ck["config"]
    if hashlib.sha256(Path(args.urdf).read_bytes()).hexdigest() != c["urdf_sha256"]:
        raise ValueError("URDF differs from saved physics contract")
    torch.set_float32_matmul_precision("high" if conf.get("tf32") else "highest")
    model = StateWorldModel(
        c["context_dim"],
        hidden_layers=conf["wm_hidden_dims"],
        com_indices=c["com_indices"],
    ).to(args.device)
    model.load_state_dict(ck["model"])
    replay = Replay(conf["capacity_steps"], args.device)
    replay.load_state_dict(ck["replay"])
    torch.manual_seed(114)
    seq = replay.sample(args.batch, 8)
    fk = Kinematics(args.urdf, c["joint_names"]).to(args.device)

    def synchronize():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def model_eval(call):
        s = seq["state"][0].detach().clone().requires_grad_(True)
        targets = seq["target"].detach().clone().requires_grad_(True)
        x = s
        loss = 0
        for j in range(8):
            x = call(x, targets[j], seq["context"][j])
            loss = loss + x.square().mean()
        grads = torch.autograd.grad(loss, (s, targets, *model.parameters()))
        return x.detach(), tuple(g.detach() for g in grads)

    def fk_eval(call):
        s = seq["state"][0].detach().clone().requires_grad_(True)
        pos, quat = call(s)
        result = torch.cat([torch.cat((pos[n], quat[n]), -1) for n in pos], -1)
        return result.detach(), (torch.autograd.grad(result.square().mean(), s)[0],)

    result = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "backend": args.backend,
        "optimizer_updates": 0,
        "components": {},
    }
    for name, module, run in [("dynamics", model, model_eval), ("fk", fk, fk_eval)]:
        compiled = torch.compile(module, backend=args.backend, fullgraph=True)
        eager = run(module)
        candidate = run(compiled)
        torch.testing.assert_close(eager[0], candidate[0], atol=1e-4, rtol=2e-3)
        for a, b in zip(eager[1], candidate[1]):
            torch.testing.assert_close(a, b, atol=1e-4, rtol=2e-3)
        times = {}
        for tag, call in [("eager", module), ("compiled", compiled)]:
            synchronize()
            start = time.monotonic()
            for _ in range(3):
                run(call)
            synchronize()
            times[tag] = (time.monotonic() - start) / 3
        result["components"][name] = {
            "passed": True,
            "output_max_abs_diff": float((eager[0] - candidate[0]).abs().max()),
            "gradient_max_abs_diff": max(
                float((a - b).abs().max()) for a, b in zip(eager[1], candidate[1])
            ),
            "forward_backward_seconds": times,
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
