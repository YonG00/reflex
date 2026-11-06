"""Isaac entry point. --mode preflight first; training requires explicit actor gate."""

import argparse
from reflex import paths
import os
import numpy as np  # Import before AppLauncher alters numpy paths.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=(
        "preflight",
        "train",
        "play",
        "contact_probe",
        "gradient_audit",
        "compile_step_check",
        "event_boundary_probe",
        "event_inventory",
        "event_condition_probe",
        "v24_short_compare",
        "actor_profile",
        "actor_timing",
        "actor_view_benchmark",
        "actor_update_equivalence",
        "v24_fusion_check",
        "actor_fusion_update_equivalence",
        "imagination_boundary_check",
        "pure_soft_survival_check",
        "actor_component_timing",
        "root_tracking_check",
        "physical_action_ranking",
        "collect_component_timing",
        "event_compact_optimization_check",
    ),
    default="preflight",
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_iterations", type=int, default=100)
parser.add_argument(
    "--max_seconds",
    type=float,
    default=None,
    help="Training wall-time budget; finish current update then save",
)
parser.add_argument(
    "--profile_joint",
    action="store_true",
    help="Profile first Actor update without changing training config",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--config",
    default=os.path.join(paths.REPO_ROOT, "configs/world_model/default.json"),
)
parser.add_argument("--checkpoint")
parser.add_argument("--migrate_event_execution", action="store_true")
parser.add_argument("--migrate_update_rate", action="store_true")
parser.add_argument("--migrate_soft_survival", action="store_true")
parser.add_argument("--aggregate_gradient_audit", action="store_true")
parser.add_argument("--migrate_imagination_boundaries", action="store_true")
parser.add_argument(
    "--reinitialize_replay",
    action="store_true",
    help="Explicit new simulator world: retain weights/optimizers, rebuild replay and collection warmup",
)
parser.add_argument(
    "--warm_start_macro", help="New macro-step experiment; reset replay/WM optimizer"
)
parser.add_argument(
    "--migrate_termination_training",
    action="store_true",
    help="Explicitly migrate separate real-feature head training to joint predicted BCE",
)
parser.add_argument(
    "--warm_start_actor", help="PPO Teacher engineering bootstrap; not full resume"
)
parser.add_argument("--output", default="logs/world_model/stage1")
parser.add_argument(
    "--enable_actor",
    action="store_true",
    help="Enable Actor updates under the selected guarded or joint schedule",
)
parser.add_argument("--save_every", type=int, default=25)
parser.add_argument("--preflight_steps", type=int, default=512)
parser.add_argument("--benchmark_steps", type=int, default=24)
parser.add_argument(
    "--benchmark_no_checkpoint",
    action="store_true",
    help="Bounded throughput trials only; discard trial weights, retain metrics",
)
parser.add_argument("--play_steps", type=int, default=1500)
parser.add_argument("--video", action="store_true")
parser.add_argument(
    "--no_debug_vis",
    action="store_true",
    help="Disable only command/sensor debug drawings",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (
    sum(
        bool(x) for x in (args.checkpoint, args.warm_start_actor, args.warm_start_macro)
    )
    > 1
):
    parser.error("Choose exactly one resume/warm-start source")
if args.checkpoint and args.warm_start_actor:
    parser.error("--checkpoint and --warm_start_actor are mutually exclusive")
if args.num_envs < 6 or args.save_every < 1:
    parser.error(
        "Use at least 6 envs for separate holdout environments and save_every >= 1"
    )
if args.benchmark_no_checkpoint and (
    args.mode != "train"
    or args.max_seconds is None
    or args.max_seconds > 600
    or args.max_iterations > 200
):
    parser.error(
        "benchmark_no_checkpoint is restricted to <=600s, <=200-iteration train trials"
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
    from .task import AnalyticTask
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
    if args.no_debug_vis:
        cfg.commands.motion.debug_vis = False
        if hasattr(cfg.scene, "contact_forces"):
            cfg.scene.contact_forces.debug_vis = False
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Save config before simulator creation, so failed preflight has evidence.
    config = TrainConfig(**json.loads(Path(args.config).read_text()))
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    if config.monitor_contacts:
        from isaaclab.sensors import ContactSensorCfg

        cfg.scene.balloon.spawn.activate_contact_sensors = True
        import xml.etree.ElementTree as ET

        robot_xml = ET.parse(cfg.scene.robot.spawn.asset_path).getroot()
        rigid_links = ["pelvis"] + [
            j.find("child").get("link")
            for j in robot_xml.findall("joint")
            if j.get("type") != "fixed"
        ]
        cfg.scene.wm_box_contacts = ContactSensorCfg(
            prim_path=cfg.scene.balloon.prim_path,
            filter_prim_paths_expr=[
                cfg.scene.robot.prim_path + "/" + name for name in rigid_links
            ],
            history_length=cfg.decimation,
            max_contact_data_count_per_prim=32,
            update_period=0.0,
            debug_vis=False,
        )
    elif hasattr(cfg.scene, "wm_box_contacts"):
        cfg.scene.wm_box_contacts = None
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
        adapter = IsaacAdapter(
            env,
            urdf,
            monitor_contacts=config.monitor_contacts,
            learn_external_events=config.learn_external_events,
            parity_every=config.real_parity_every,
            cache_context=config.cache_physical_context,
            native_collection_inputs=config.native_collection_inputs,
        )
        if (
            not config.monitor_contacts
            and "wm_box_contacts" in env.unwrapped.scene.sensors
        ):
            raise RuntimeError("Optional box monitoring sensor was not disabled")
        task = None
        if config.wm_kind == "state":
            task = AnalyticTask.from_environment(adapter)
            adapter.attach_task(task)
        event_scheduler = None
        if config.condition_external_events:
            from .events_v24.runtime import NativeEventScheduler

            event_scheduler = NativeEventScheduler(
                env.unwrapped, task, seed=args.seed + 24000
            )
            adapter.event_scheduler = event_scheduler
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
            task=task,
            event_scheduler=event_scheduler,
        )
        trainer.contact_filter_schema = (
            "explicit_urdf_rigid_links_v2" if config.monitor_contacts else "disabled"
        )
        trainer.profile_dir = str(out / "profiles")
        trainer.profile_next_actor = args.profile_joint
        if args.checkpoint:
            trainer.load(
                args.checkpoint,
                migrate_termination_training=args.migrate_termination_training,
                reinitialize_replay=args.reinitialize_replay,
                migrate_imagination_boundaries=args.migrate_imagination_boundaries,
                migrate_soft_survival=args.migrate_soft_survival,
                migrate_event_execution=args.migrate_event_execution,
                migrate_update_rate=args.migrate_update_rate,
            )
        elif args.warm_start_macro:
            trainer.warm_start_macro(args.warm_start_macro)
        elif args.warm_start_actor:
            trainer.warm_start_actor(args.warm_start_actor)
        manifest = adapter.manifest()
        manifest["actor_initialization"] = trainer.initialization
        manifest["imagination_start_policy"] = (
            "all_valid_states_uniform_with_replacement_v2"
        )
        manifest["learn_external_events"] = config.learn_external_events
        manifest["event_scheduler"] = (
            event_scheduler.specification() if event_scheduler is not None else None
        )
        manifest["event_view_execution"] = {
            "kind": "minimal_root_joint"
            if config.minimal_event_view
            else "full_task_view",
            "source_sha256": hashlib.sha256(
                Path(__file__)
                .with_name("events_v24")
                .joinpath("fast_view.py")
                .read_bytes()
            ).hexdigest()
            if config.minimal_event_view
            else None,
        }
        manifest["contact_monitoring_enabled"] = config.monitor_contacts
        manifest["contact_filter_schema"] = trainer.contact_filter_schema
        manifest["contact_filter_migration"] = trainer.contact_filter_migration
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
        parity = adapter.parity(b, critic=config.privileged_critic)
        (out / "parity.json").write_text(json.dumps(parity, indent=2))
        if args.mode == "event_compact_optimization_check":
            from .events_v24.actor_profile import update_equivalence
            from .events_v24.component_timing import check

            update_equivalence(trainer, out, compact=True)
            original_compact = trainer.event_scheduler.compact_inputs
            try:
                for enabled, label in ((False, "baseline"), (True, "compact")):
                    trainer.event_scheduler.compact_inputs = enabled
                    print(json.dumps(check(trainer, out / label)), flush=True)
            finally:
                trainer.event_scheduler.compact_inputs = original_compact
            return
        if args.mode == "collect_component_timing":
            from .events_v24.collect_timing import check

            print(
                json.dumps(check(trainer, adapter, out, steps=args.benchmark_steps)),
                flush=True,
            )
            return
        if args.mode == "physical_action_ranking":
            result = rank_actions(adapter, trainer, horizon=1, min_pairs=64)
            (out / "physical_action_ranking.json").write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(result), flush=True)
            return
        if args.mode == "root_tracking_check":
            from .events_v24.root_tracking import check

            print(json.dumps(check(trainer, out)), flush=True)
            return
        if args.mode == "actor_component_timing":
            from .events_v24.component_timing import check

            print(json.dumps(check(trainer, out)), flush=True)
            return
        if args.mode == "pure_soft_survival_check":
            from .events_v24.boundary_check import check_soft

            print(json.dumps(check_soft(trainer, out)), flush=True)
            return
        if args.mode == "imagination_boundary_check":
            from .events_v24.boundary_check import check

            print(json.dumps(check(trainer, out)), flush=True)
            return
        if args.mode == "actor_fusion_update_equivalence":
            from .events_v24.actor_profile import update_equivalence

            update_equivalence(trainer, out, fusion=True)
            return
        if args.mode == "v24_fusion_check":
            from .events_v24.fused_step import check

            check(trainer, out)
            return
        if args.mode == "actor_update_equivalence":
            from .events_v24.actor_profile import update_equivalence

            update_equivalence(trainer, out)
            return
        if args.mode == "actor_view_benchmark":
            from .events_v24.actor_profile import benchmark_views

            benchmark_views(trainer, out)
            return
        if args.mode == "actor_timing":
            from .events_v24.actor_profile import run

            run(trainer, out, profile=False)
            return
        if args.mode == "actor_profile":
            from .events_v24.actor_profile import run

            run(trainer, out)
            return
        if args.mode == "v24_short_compare":
            from .events_v24.compare import run

            run(adapter, trainer, out)
            return
        if args.mode == "event_condition_probe":
            from .events_v24.probe import check

            result = check(adapter, trainer)
            (out / "event_condition_probe.json").write_text(
                json.dumps(result, indent=2)
            )
            print(json.dumps(result), flush=True)
            return
        if args.mode == "event_inventory":
            from .event_inventory import inspect

            result = inspect(adapter, trainer)
            (out / "event_inventory.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result["after_one_step"]["events"], indent=2), flush=True)
            return
        if args.mode == "event_boundary_probe":
            from .event_probe import check

            result = check(adapter, trainer, trace_path=out / "event_trace.json")
            (out / "event_boundary_probe.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps({k: v for k, v in result.items() if k != "transitions"}),
                flush=True,
            )
            return
        if args.mode == "compile_step_check":
            if not args.checkpoint:
                raise ValueError("compile_step_check requires --checkpoint")
            from .compile_benchmark import check

            result = check(trainer)
            (out / "compile_step_check.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return
        if args.mode == "gradient_audit":
            if not args.checkpoint:
                raise ValueError("gradient_audit requires --checkpoint")
            from .diagnostics import gradient_audit

            before = (trainer.actor_updates, trainer.model_updates)
            compiled = (trainer._dynamics, trainer._fk)
            result = gradient_audit(
                trainer, per_reward=not args.aggregate_gradient_audit
            )
            assert before == (trainer.actor_updates, trainer.model_updates)
            assert trainer._dynamics is compiled[0] and trainer._fk is compiled[1]
            result["optimizer_updates"] = 0
            (out / "gradient_audit.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            return
        if args.mode == "contact_probe":
            if not config.monitor_contacts:
                raise ValueError(
                    "contact_probe requires explicitly enabled optional contact monitoring"
                )
            from .contact_probe import check

            result = check(adapter, trainer)
            (out / "contact_probe.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            if not result["passed"]:
                raise RuntimeError(
                    "Controlled robot/ground contact filtering check failed"
                )
            return
        if args.mode == "play":
            if not args.checkpoint:
                raise ValueError("play requires --checkpoint")
            # Adapter uses unwrapped environment for capture; video wrapper must
            # receive the step for recording, so supply a dispatch hook.
            adapter.step_environment = env.step
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            from .experience import ExperienceMetrics

            evaluation = ExperienceMetrics(monitor_contacts=config.monitor_contacts)
            play_started = time.monotonic()
            rewards = []
            terminated = 0
            strict_envs = torch.zeros(
                args.num_envs, dtype=torch.bool, device=args.device
            )
            strict_steps = 0
            with torch.no_grad():
                for _ in range(args.play_steps):
                    b = adapter.snapshot()
                    raw = trainer.act(b)
                    target, blend = trainer.targets(b, raw)
                    tr = adapter.step(raw, target, blend)
                    evaluation.update(tr)
                    rewards.append(float(tr["reward_terms"].sum(-1).mean()))
                    terminated += int(tr["terminated"].sum())
                    # Original strict success is >=30 consecutive frames (0.6s).
                    strict = adapter.pre_reset["task_history_after_reward"][:, -1] >= 30
                    strict_envs |= strict
                    strict_steps += int(strict.sum())
            summary = {
                "envs": args.num_envs,
                "vector_steps": args.play_steps,
                "mean_step_task_reward": sum(rewards) / len(rewards),
                "physical_terminations": terminated,
                "strict_success_envs": int(strict_envs.sum()),
                "strict_success_env_steps": strict_steps,
                "actor_updates_in_checkpoint": trainer.actor_updates,
                "evaluation_seed": args.seed,
                "evaluation_rng_reset": True,
            }
            summary["wall_seconds"] = time.monotonic() - play_started
            summary["fresh_episode_metrics"] = evaluation.metrics()
            summary["termination_names"] = adapter.contract["termination_names"]
            (out / "play_summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
            return
        if args.mode == "preflight":
            old = config.collect_steps
            config.collect_steps = args.preflight_steps
            trainer.collect(adapter)
            config.collect_steps = old
            adapter.parity(adapter.snapshot(), critic=config.privileged_critic)
            counts = {
                str(h): len(trainer.replay.indices(h, starts=True))
                for h in (1, 4, 8, 16, 32)
            }
            result = {
                "passed": len(trainer.replay.indices(config.model_window)) > 0
                and len(trainer.replay.start_indices()) > 0,
                "parity": parity,
                "postlaunch_windows": counts,
                "imagined_start_snapshots": len(trainer.replay.start_indices()),
                "supervision_window": config.model_window,
                "real_transitions": trainer.real_transitions,
                "actor_updated": False,
                "task_parity_max": {
                    n: float(v)
                    for n, v in zip(adapter.reward_names, adapter.task_parity_max)
                },
            }
            result["experience"] = trainer.experience.metrics()
            if config.monitor_contacts:
                result["passed"] &= (
                    result["experience"]["experience/contact_measured_fraction"] == 1.0
                )
            if config.privileged_critic:
                result["passed"] &= (
                    result["experience"]["experience/termination_reasons_measured"]
                    == 1.0
                )
            (out / "preflight.json").write_text(json.dumps(result, indent=2))
            trainer.save(out / "preflight.pt")
            if not result["passed"]:
                raise RuntimeError(
                    "Preflight has no valid WM supervision windows or imagination snapshots"
                )
            print(json.dumps(result), flush=True)
            return
        writer = SummaryWriter(str(out / "tensorboard"))
        start = time.monotonic()
        initial_steps = trainer.real_transitions
        try:
            while trainer.iteration < args.max_iterations:
                if (
                    args.max_seconds is not None
                    and time.monotonic() - start >= args.max_seconds
                ):
                    break
                trainer.replay_seconds = 0.0
                collect_start = timestamp(args.device)
                metrics = trainer.collect(adapter)
                metrics["time/collect_seconds"] = timestamp(args.device) - collect_start
                metrics["schedule/update_due"] = float(trainer.updates_due())
                try:
                    if not trainer.updates_due():
                        raise ValueError("Not a scheduled update step")
                    trainer.sample(config.model_window)
                except ValueError:
                    metrics["warmup/insufficient_windows"] = float(
                        bool(trainer.updates_due())
                    )
                    metrics["schedule/update_due"] = float(trainer.updates_due())
                else:
                    metrics["schedule/model_lr"] = trainer.schedule_lr()
                    metrics.update(trainer.fit_model())
                    validation_start = timestamp(args.device)
                    if trainer.validation_due():
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
                        and config.schedule != "joint_reproduction"
                        and trainer.iteration >= config.warmup_iterations
                        and trainer.basic_ready(metrics)
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
                    approved = args.enable_actor and trainer.ready(
                        metrics, ranking["passed"]
                    )
                    metrics["gate/actor_enabled"] = float(approved)
                    if approved:
                        try:
                            trainer.sample(starts=True)
                        except ValueError:
                            metrics["gate/no_imagination_starts"] = 1.0
                        else:
                            from .diagnostics import gradient_audit, diagnostic_due

                            audit_due, audit_budget_skip = diagnostic_due(
                                trainer, time.monotonic() - start, args.max_seconds
                            )
                            metrics["diagnostic/skipped_time_budget"] = float(
                                audit_budget_skip
                            )
                            if audit_due:
                                metrics.update(gradient_audit(trainer))
                                trainer.last_diagnostic_seconds = metrics[
                                    "time/gradient_diagnostic_seconds"
                                ]
                                metrics["coverage/wm8_windows"] = len(
                                    trainer.replay.indices(config.model_window)
                                )
                                metrics["coverage/actor_start_snapshots"] = len(
                                    trainer.replay.start_indices()
                                )
                            metrics.update(trainer.update_actor())
                trainer.iteration += 1
                if (
                    trainer.iteration % args.save_every == 0
                    and not args.benchmark_no_checkpoint
                ):
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
                        "progress/wm_gradient_updates": trainer.model_updates,
                        "progress/termination_gradient_updates": trainer.termination_updates,
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
            if args.benchmark_no_checkpoint:
                (out / "benchmark_end.json").write_text(
                    json.dumps(
                        {
                            "completed": True,
                            "checkpoint_saved": False,
                            "trial_weights_discarded": True,
                            "iteration": trainer.iteration,
                            "actor_updates": trainer.actor_updates,
                            "model_updates": trainer.model_updates,
                            "termination_updates": trainer.termination_updates,
                        },
                        indent=2,
                    )
                )
            else:
                trainer.save(out / "checkpoint_final.pt")
        finally:
            writer.close()
    finally:
        if "manifest" in locals():
            manifest.update(adapter.manifest())
            (out / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False)
            )
        env.close()


failed = False
try:
    main()
except BaseException:
    failed = True
    import traceback
    from pathlib import Path

    failure = traceback.format_exc()
    print(failure, flush=True)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "failure.txt").write_text(failure)
finally:
    try:
        app.close()
    finally:
        if failed:
            raise SystemExit(1)
