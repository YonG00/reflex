"""Explicit batch-only checkpoint migration; tensors and optimizers are untouched."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("source", type=Path)
p.add_argument("target", type=Path)
p.add_argument("--batch", type=int, required=True)
a = p.parse_args()
assert a.source.is_file() and not a.target.exists()
s = torch.load(a.source, map_location="cpu", weights_only=False)
required = {
    "config",
    "model",
    "actor",
    "critic",
    "model_opt",
    "actor_opt",
    "critic_opt",
    "learning_contract",
}
assert required <= s.keys(), "Need a complete joint-training checkpoint"
keys = ("batch", "actor_batch", "actor_microbatch")
old = {k: s["config"][k] for k in keys}
assert 0 < a.batch < min(old.values()), "Explicit reduction only"
change = {
    "source": str(a.source.resolve()),
    "source_sha256": hashlib.sha256(a.source.read_bytes()).hexdigest(),
    "from": old,
    "to": {k: a.batch for k in keys},
    "weights_optimizers_normalizers_unchanged": True,
}
for k in keys:
    s["config"][k] = a.batch
s.setdefault("initialization", {})["explicit_batch_reduction"] = change
# Network/replay schemas, environment count and learning contract stay unchanged.
a.target.parent.mkdir(parents=True, exist_ok=True)
torch.save(s, a.target)
a.target.with_suffix(".migration.json").write_text(json.dumps(change, indent=2))
print(json.dumps(change), flush=True)
