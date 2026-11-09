"""Wait for one known training worker, then evaluate its final checkpoint once."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("training_pid", type=int)
p.add_argument("run", type=Path)
a = p.parse_args()
status = a.run.with_suffix(".post_eval.json")
root = Path(__file__).resolve().parents[2]


def report(state, **fields):
    status.write_text(
        json.dumps({"state": state, "run": str(a.run), **fields}, indent=2)
    )


try:
    report("waiting_for_training")
    deadline = time.monotonic() + 3000
    while True:
        try:
            os.kill(a.training_pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Training worker did not exit before evaluation deadline"
            )
        time.sleep(10)
    phase = json.loads((a.run / "training_phases.json").read_text())
    checkpoint = a.run / "checkpoint_final.pt"
    if phase["state"] != "completed" or not checkpoint.is_file():
        raise RuntimeError("Training did not produce a completed final checkpoint")
    output = a.run.with_name(a.run.name + "_play15s")
    report("evaluating", checkpoint=str(checkpoint), output=str(output))
    with a.run.with_suffix(".post_eval.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/local/run_diagnostic_group.py"),
                "--timeout",
                "900",
                "--grace",
                "10",
                "--",
                "bash",
                str(root / "scripts/local/eval_wm_video.sh"),
                str(checkpoint),
                "--seconds",
                "15",
                "--seed",
                "42",
                "--num-envs",
                "1",
                "--output",
                str(output),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not (output / "play_summary.json").is_file():
        raise RuntimeError("Independent evaluation summary missing")
    report("completed", checkpoint=str(checkpoint), output=str(output))
except Exception as exc:
    report("failed", error=repr(exc))
    raise
