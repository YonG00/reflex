"""Isaac entry point. --mode preflight first; training requires explicit actor gate."""

import argparse
from reflex import paths
import os
import numpy as np  # Import before AppLauncher alters numpy paths.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=("preflight", "train", "play"), default="preflight"
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=100)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--config",
    default=os.path.join(paths.REPO_ROOT, "configs/world_model/default.json"),
)
parser.add_argument("--checkpoint")
parser.add_argument(
    "--warm_start_actor", help="PPO Teacher engineering bootstrap; not full resume"
)
parser.add_argument("--output", default="logs/world_model/stage1")
parser.add_argument(
    "--enable_actor",
    action="store_true",
    help="Attempt gated actor updates after WM warmup; never bypasses validation",
)
parser.add_argument("--save_every", type=int, default=25)
parser.add_argument("--preflight_steps", type=int, default=512)
parser.add_argument("--play_steps", type=int, default=1500)
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint and args.warm_start_actor:
    parser.error("--checkpoint and --warm_start_actor are mutually exclusive")
if args.num_envs < 6 or args.save_every < 1:
    parser.error(
        "Use at least 6 envs for separate holdout environments and save_every >= 1"
    )
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app


def main():
    import dataclasses
    import hashlib
    import json
    import random
    import subprocess
    import time
    from pathlib import Path
    import gymnasium as gym
    import torch
    from torch.utils.tensorboard import SummaryWriter
    import whole_body_tracking.tasks  # noqa: F401 - task registration
    from .oracle_env import OracleEnvCfg, register
    from .isaac_adapter import IsaacAdapter
    from .prior import MotionPrior
    from .trainer import Trainer, TrainConfig
    from .probes import rank_actions
    from .timing import timestamp

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    register()
    cfg = OracleEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.commands.motion.motion_file = paths.MOTION_FILE
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Save config before simulator creation, so failed preflight has evidence.
    config = TrainConfig(**json.loads(Path(args.config).read_text()))
    env = gym.make(
        "Reflex-Teacher-Oracle-G1-v0",
        cfg=cfg,
        render_mode="rgb_array" if args.video else None,
    )
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(out / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=args.play_steps,
            disable_logger=True,
        )
    try:
        env.reset(seed=args.seed)
        # Parse exactly the asset that the environment imported, rather than a
        # convenient local URDF that could have different link/joint frames.
        urdf = cfg.scene.robot.spawn.asset_path
        adapter = IsaacAdapter(env, urdf)
        prior = MotionPrior(
            os.environ.get("AMP_DATASET", paths.AMP_DATASET),
            args.device,
            batch=min(1024, config.batch * 4),
        )
        trainer = Trainer(
            adapter.bridge,
            adapter.action_map,
            prior,
            adapter.contract["context_dim"],
            6,
            adapter.reward_names,
            adapter.contract,
            args.device,
            config,
        )
        if args.checkpoint:
            trainer.load(args.checkpoint)
        elif args.warm_start_actor:
            trainer.warm_start_actor(args.warm_start_actor)
        manifest = adapter.manifest()
        manifest["actor_initialization"] = trainer.initialization
        manifest.update(
            {
                "config": dataclasses.asdict(config),
                "args": vars(args),
                "git_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, text=True
                ).strip(),
                "amp_sha256": hashlib.sha256(
                    Path(os.environ.get("AMP_DATASET", paths.AMP_DATASET)).read_bytes()
                ).hexdigest(),
                "simulator_state_restored_on_resume": False,
            }
        )
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False)
        )
        b = adapter.snapshot()
        parity = adapter.parity(b)
        (out / "parity.json").write_text(json.dumps(parity, indent=2))
        if args.mode == "play":
            if not args.checkpoint:
                raise ValueError("play requires --checkpoint")
            # Adapter uses unwrapped environment for capture; video wrapper must
            # receive the step for recording, so supply a dispatch hook.
            adapter.step_environment = env.step
            with torch.no_grad():
                for _ in range(args.play_steps):
                    b = adapter.snapshot()
                    raw = trainer.act(b)
                    target, blend = trainer.targets(b, raw)
                    adapter.step(raw, target, blend)
            return
        if args.mode == "preflight":
            old = config.collect_steps
            config.collect_steps = args.preflight_steps
            trainer.collect(adapter)
            config.collect_steps = old
            adapter.parity(adapter.snapshot())
            counts = {
                str(h): len(trainer.replay.indices(h, starts=True))
                for h in (1, 4, 8, 16, 32)
            }
            result = {
                "passed": int(counts["32"]) > 0,
                "parity": parity,
                "postlaunch_windows": counts,
                "real_transitions": trainer.real_transitions,
                "actor_updated": False,
            }
            (out / "preflight.json").write_text(json.dumps(result, indent=2))
            trainer.save(out / "preflight.pt")
            if not result["passed"]:
                raise RuntimeError("Preflight has no valid 32-step post-launch windows")
            print(json.dumps(result), flush=True)
            return
        writer = SummaryWriter(str(out / "tensorboard"))
        start = time.monotonic()
        initial_steps = trainer.real_transitions
        try:
            while trainer.iteration < args.max_iterations:
                trainer.replay_seconds = 0.0
                collect_start = timestamp(args.device)
                metrics = trainer.collect(adapter)
                metrics["time/collect_seconds"] = timestamp(args.device) - collect_start
                try:
                    trainer.sample(config.model_window)
                except ValueError:
                    metrics["warmup/insufficient_windows"] = 1.0
                else:
                    metrics.update(trainer.fit_model())
                    validation_start = timestamp(args.device)
                    metrics.update(trainer.validate())
                    metrics["time/validation_seconds"] = (
                        timestamp(args.device) - validation_start
                    )
                    ranking = {
                        "passed": False,
                        "reason": "actor updates not enabled / warmup",
                    }
                    if (
                        args.enable_actor
                        and trainer.iteration >= config.warmup_iterations
                    ):
                        ranking_start = timestamp(args.device)
                        ranking = rank_actions(adapter, trainer)
                        metrics["time/ranking_seconds"] = (
                            timestamp(args.device) - ranking_start
                        )
                        (out / f"ranking_{trainer.iteration:05d}.json").write_text(
                            json.dumps(ranking, indent=2)
                        )
                    metrics["gate/ranking_passed"] = float(ranking["passed"])
                    approved = trainer.ready(metrics, ranking["passed"])
                    metrics["gate/actor_enabled"] = float(approved)
                    if approved:
                        try:
                            trainer.sample(config.horizon, starts=True)
                        except ValueError:
                            metrics["gate/no_imagination_starts"] = 1.0
                        else:
                            metrics.update(trainer.update_actor())
                trainer.iteration += 1
                if trainer.iteration % args.save_every == 0:
                    save_start = timestamp(args.device)
                    trainer.save(out / f"checkpoint_{trainer.iteration:05d}.pt")
                    metrics["time/checkpoint_seconds"] = (
                        timestamp(args.device) - save_start
                    )
                metrics.update(
                    {
                        "progress/real_transitions": trainer.real_transitions,
                        "progress/probe_transitions": trainer.probe_transitions,
                        "progress/collect_transitions": trainer.real_transitions
                        - trainer.probe_transitions,
                        "progress/actor_updates": trainer.actor_updates,
                        "time/replay_sample_seconds_inclusive": trainer.replay_seconds,
                        "progress/wall_seconds": time.monotonic() - start,
                        "progress/end_to_end_transitions_per_second": (
                            trainer.real_transitions - initial_steps
                        )
                        / (time.monotonic() - start),
                    }
                )
                for k, v in metrics.items():
                    writer.add_scalar(k, v, trainer.iteration)
                writer.flush()
                record = {"iteration": trainer.iteration, **metrics}
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
            trainer.save(out / "checkpoint_final.pt")
        finally:
            writer.close()
    finally:
        env.close()


try:
    main()
finally:
    app.close()
