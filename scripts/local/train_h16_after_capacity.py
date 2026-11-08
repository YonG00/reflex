"""Run authorized fresh WM warmup + joint30 only after the exact capacity gate."""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def validate(report, config):
    if not report.get("passed") or len(report.get("rows", [])) != 4:
        raise ValueError("Four complete capacity updates required")
    expected = {"batch": 4096, "actor_batch": 4096, "actor_microbatch": 4096}
    if (
        report.get("effective_batches") != expected
        or report.get("effective_horizon") != 16
    ):
        raise ValueError("Capacity batch/Horizon mismatch")
    if any(
        config.get(k) != v
        for k, v in {
            **expected,
            "horizon": 16,
            "model_window": 8,
            "wm_only_rounds": 500,
        }.items()
    ):
        raise ValueError("Fresh training configuration mismatch")
    if any(
        config.get(k, False)
        for k in ("block_event_writes", "fused_actor_carry", "fuse_actor_core")
    ):
        raise ValueError("Unaccepted execution candidate enabled")
    for row in report["rows"]:
        for section in ("model", "actor"):
            if any(
                isinstance(v, (int, float)) and not math.isfinite(v)
                for v in row[section].values()
            ):
                raise ValueError("Nonfinite capacity metrics")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capacity-output", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    status_path = output.with_suffix(".job.json")
    if status_path.exists():
        raise FileExistsError(status_path)
    status = {
        "state": "waiting_capacity",
        "training_output": str(output),
        "from_scratch": True,
    }

    def save(**values):
        status.update(values)
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(status_path)
        print(json.dumps(status), flush=True)

    def run(command, logfile, stage):
        with logfile.open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            save(
                state=stage,
                supervisor_pid=process.pid,
                command=command,
                log=str(logfile),
            )
            return process.wait()

    save()
    try:
        exit_file = Path(str(args.capacity_output) + ".exit")
        deadline = time.monotonic() + 900
        while not exit_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Capacity test did not finish")
            time.sleep(5)
        report = json.loads((args.capacity_output / "capacity_check.json").read_text())
        config = json.loads(args.config.read_text())
        validate(report, config)
        output.mkdir()
        supervisor = [
            sys.executable,
            str(ROOT / "scripts/local/run_diagnostic_group.py"),
        ]
        command = supervisor + [
            "--timeout",
            "6000",
            "--grace",
            "20",
            "--",
            "bash",
            str(ROOT / "scripts/train_stage1_world_model.sh"),
            "--mode",
            "train",
            "--num_envs",
            "4096",
            "--config",
            str(args.config.resolve()),
            "--enable_actor",
            "--joint_max_seconds",
            "1800",
            "--max_iterations",
            "100000",
            "--save_every",
            "200",
            "--seed",
            "42",
            "--output",
            str(output),
            "--no_debug_vis",
        ]
        # No checkpoint/warm-start argument: all trainable models start fresh.
        code = run(command, output / "train.log", "training_running")
        phases = json.loads((output / "training_phases.json").read_text())
        checkpoint = output / "checkpoint_final.pt"
        if (
            code
            or (output / "failure.txt").exists()
            or phases.get("state") != "completed"
            or phases.get("joint_wall_seconds", 0) < 1800
            or not checkpoint.is_file()
        ):
            raise RuntimeError(f"Training incomplete, exit={code}")
        video = output.parent / (output.name + "_play15s")
        command = supervisor + [
            "--timeout",
            "240",
            "--grace",
            "10",
            "--",
            "bash",
            str(ROOT / "scripts/local/eval_wm_video.sh"),
            str(checkpoint),
            "--seconds",
            "15",
            "--seed",
            "42",
            "--num-envs",
            "1",
            "--output",
            str(video),
        ]
        code = run(command, output / "eval.log", "evaluating")
        if code:
            raise RuntimeError(f"Video evaluation failed, exit={code}")
        save(state="completed", checkpoint=str(checkpoint), video_output=str(video))
    except BaseException as error:
        save(state="failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
