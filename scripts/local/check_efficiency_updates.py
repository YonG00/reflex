#!/usr/bin/env python3
"""CPU synthetic WM + Actor/Critic Adam equivalence against a fixed Git ref.

Uses the repository's small test fixture; does not validate PhysX or CUDA.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="63b3ec8")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "tests"))
    import test_repaired_joint as fixture
    from reflex.world_model.trainer import Trainer

    def compare(x, y):
        if isinstance(x, torch.Tensor):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
            return float((x - y).abs().max()) if x.numel() else 0.0
        if isinstance(x, dict):
            assert x.keys() == y.keys()
            return max((compare(v, y[k]) for k, v in x.items()), default=0.0)
        if isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            return max((compare(a, b) for a, b in zip(x, y)), default=0.0)
        assert x == y
        return 0.0

    with tempfile.TemporaryDirectory(prefix="reflex_efficiency_") as directory:
        tmp = Path(directory)

        def baseline(file):
            name = "reflex.world_model._efficiency_baseline_" + file
            path = tmp / (file + ".py")
            path.write_bytes(
                subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{args.baseline_ref}:source/reflex/reflex/world_model/{file}.py",
                    ],
                    cwd=root,
                )
            )
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        old, old_joint = baseline("trainer"), baseline("joint")
        # Populate expert data before the two equally seeded constructions.
        fixture.repaired(tmp)
        try:
            fixture.Trainer = old.Trainer
            reference = fixture.repaired(tmp)
            fixture.Trainer = Trainer
            actual = fixture.repaired(tmp)
            actual.cfg.synchronize_replay_timing = False
        finally:
            fixture.Trainer = Trainer
        report = {
            "baseline_ref": args.baseline_ref,
            "scope": "CPU synthetic; not PhysX/CUDA",
        }
        for obj in (reference, actual):
            torch.manual_seed(934)
            obj.fit_model()
        for label in ("after_wm", "after_actor"):
            if label == "after_actor":
                torch.manual_seed(735)
                old_joint._update_actor(reference)
                torch.manual_seed(735)
                actual.update_actor()
            for name in (
                "model",
                "termination",
                "actor",
                "critic",
                "model_opt",
                "termination_opt",
                "actor_opt",
                "critic_opt",
            ):
                a, b = getattr(reference, name), getattr(actual, name)
                report[label + "/" + name + "_max_abs_error"] = compare(
                    a.state_dict(), b.state_dict()
                )
                if isinstance(a, torch.nn.Module):
                    report[label + "/" + name + "_gradient_max_abs_error"] = max(
                        (
                            compare(x.grad, y.grad)
                            for x, y in zip(a.parameters(), b.parameters())
                        ),
                        default=0.0,
                    )
        report["passed"] = True
    payload = json.dumps(report, indent=2)
    if args.out:
        with args.out.open("x") as f:
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
