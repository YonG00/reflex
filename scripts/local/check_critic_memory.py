"""GPU frozen-Critic forward/VJP/Adam check at the exact H+1 by Actor batch."""

import argparse
import json
from pathlib import Path

import torch
from reflex.world_model.critic_values import evaluate
from reflex.world_model.models import mlp

p = argparse.ArgumentParser()
p.add_argument("checkpoint", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
assert not a.output.exists()
state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
c = state["config"]
weights = state["critic"]
dim = weights["0.weight"].shape[1]
critic = mlp(dim, 1, c["hidden"], layers=c["critic_hidden_dims"]).cuda()
critic.load_state_dict(weights)
critic.requires_grad_(False)
torch.backends.cuda.matmul.allow_tf32 = c["tf32"]
torch.manual_seed(42)
base = torch.randn(c["horizon"] + 1, c["actor_batch"], dim, device="cuda")
# A shared trainable input models the upstream Actor/WM gradient boundary.
parameter = torch.nn.Parameter(torch.randn(dim, device="cuda") * 0.01)
initial = parameter.detach().clone()
expected = None
rows = {}
for mode in ("full", "chunk_checkpoint"):
    parameter.data.copy_(initial)
    optimizer = torch.optim.Adam([parameter], lr=c["lr"], betas=(0.9, c["actor_beta2"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    obs = base + parameter
    value = evaluate(critic, obs, 4096 if mode != "full" else 0, mode != "full")
    loss = value.square().mean()
    loss.backward()
    grad = parameter.grad.detach().clone()
    optimizer.step()
    end.record()
    torch.cuda.synchronize()
    actual = (value.detach().cpu(), grad.cpu(), parameter.detach().cpu().clone())
    if expected is None:
        expected = actual
    else:
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y)
    rows[mode] = {
        "seconds": start.elapsed_time(end) / 1000,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gradient_norm": float(grad.norm()),
    }
    del obs, value, loss, grad, optimizer
    parameter.grad = None
r = {
    "passed": True,
    "shape": [c["horizon"] + 1, c["actor_batch"], dim],
    "scope": "actual checkpoint Critic, frozen forward/input gradient and Adam; full PhysX capacity still required",
    "variants": rows,
}
a.output.write_text(json.dumps(r, indent=2))
print(json.dumps(r), flush=True)
