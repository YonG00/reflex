"""Fresh full-vs-compact WM ablation on one frozen real Replay; Actor stays fixed."""

import copy
import dataclasses
import gc
import json
import random
import weakref
from pathlib import Path

import numpy as np
import torch

from .compact_context import ContextSpec, audit_randomization
from .events_v24.evaluation import evaluate
from .replay import Replay
from .timing import timestamp
from .trainer import Trainer


def seed(value, scheduler):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if str(scheduler.device).startswith("cuda"):
        torch.cuda.manual_seed_all(value)
    scheduler.rng = (
        torch.Generator(device=scheduler.device).manual_seed(value).get_state()
    )


def cpu_state(obj):
    if obj is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in obj.state_dict().items()}


def run(
    source, adapter, output, rounds=500, collect_steps=256, validate_every=25, batch=64
):
    if source.context_spec is not None or rounds < 1 or validate_every < 1:
        raise ValueError(
            "Use full-context source and positive ablation rounds/validation interval"
        )
    out = Path(output)
    if (out / "context_ablation.json").exists():
        raise FileExistsError("Do not overwrite an existing context ablation")
    audit = audit_randomization(adapter.env)
    spec = ContextSpec.from_contract(source.contract)
    # Fresh collection uses the already authorized source policy; no WM or Actor updates.
    for _ in range(max(0, collect_steps - source.replay.count)):
        source.collect(adapter)
    frozen = source.replay.state_dict()
    if frozen["count"] < source.cfg.model_window:
        raise ValueError("Insufficient frozen transitions")
    torch.save(frozen, out / "common_replay.pt")
    fixed = {
        n: cpu_state(getattr(source, n))
        for n in ("actor", "critic", "obs_norm", "critic_norm")
    }
    amp = copy.deepcopy(source.prior.amp_state_dict())
    # Release source GPU ring. Both trials now start over identical fixed GPU overhead.
    source.replay = None
    gc.collect()
    if str(source.device).startswith("cuda"):
        torch.cuda.empty_cache()
    report = {
        "audit": audit,
        "context": spec.specification(),
        "rounds": rounds,
        "gradient_updates_per_round": source.cfg.model_steps,
        "shared_data_frames": frozen["count"],
        "initialization": "fresh WM and fresh optimizers; shared hidden layers and termination initialization",
        "actor_updates": 0,
        "source_checkpoint_weights_modified": False,
        "results": {},
    }
    templates = None
    try:
        for tag, mode in [("full765", "full"), ("compact", "stage1_dr_v1")]:
            folder = out / tag
            folder.mkdir(exist_ok=False)
            cfg = dataclasses.replace(
                source.cfg,
                wm_context_mode=mode,
                save_replay=False,
                wm_mlp_precision="fp32",
                compile_wm_training_step=False,
                wm_compile_mode="default",
            )
            task = copy.deepcopy(source.task)
            scheduler = copy.deepcopy(source.event_scheduler)
            scheduler.task = task
            seed(28001, scheduler)
            source.prior.load_amp_state_dict(amp)
            seed(28001, scheduler)
            trial_contract = copy.deepcopy(source.contract)
            if mode != "full":
                trial_contract["wm_context_audit"] = {
                    **copy.deepcopy(audit),
                    "context": spec.specification(),
                }
            memory_before = (
                torch.cuda.memory_allocated()
                if str(source.device).startswith("cuda")
                else None
            )
            t = Trainer(
                source.bridge,
                source.action_map,
                source.prior,
                source.contract["context_dim"],
                6,
                source.reward_names,
                trial_contract,
                source.device,
                cfg,
                task=task,
                event_scheduler=scheduler,
            )
            for n, state in fixed.items():
                getattr(t, n).load_state_dict(state)
            if templates is None:
                templates = {
                    "model": cpu_state(t.model),
                    "termination": cpu_state(t.termination),
                }
            else:
                for name, states in templates.items():
                    if states is None:
                        continue
                    target = getattr(t, name).state_dict()
                    for k, v in states.items():
                        if k in target and target[k].shape == v.shape:
                            target[k].copy_(v.to(source.device))
                    getattr(t, name).load_state_dict(target)
            if mode == "full":
                t.replay.load_state_dict(frozen)
            else:
                for j in range(frozen["count"]):
                    ix = (
                        (j + frozen["pointer"]) % frozen["capacity"]
                        if frozen["count"] == frozen["capacity"]
                        else j
                    )
                    t.replay.add(
                        {k: v[ix].to(source.device) for k, v in frozen["data"].items()}
                    )
            seed(28002, scheduler)
            t.fit_normalizers()
            # WM/termination norms are fitted on training rows. Preserve the source
            # policy function instead of accidentally changing its observation norm.
            t.obs_norm.load_state_dict(fixed["obs_norm"])
            t.critic_norm.load_state_dict(fixed["critic_norm"])
            t.initialization = {
                "kind": "fresh_context_ablation_wm",
                "representation": mode,
                "source_actor_unchanged": True,
                "source_wm_weights_loaded": False,
            }
            records = []
            row = {"round": 0, **evaluate(t, batch=batch)}
            records.append(row)
            for j in range(rounds):
                seed(28100 + j, scheduler)
                if str(t.device).startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                start = timestamp(t.device)
                metrics = t.fit_model()
                elapsed = timestamp(t.device) - start
                record = {"round": j + 1, "wm_round_wall_seconds": elapsed, **metrics}
                if str(t.device).startswith("cuda"):
                    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                if (j + 1) % validate_every == 0 or j + 1 == rounds:
                    seed(28900, scheduler)
                    record.update(evaluate(t, batch=batch))
                records.append(record)
                with (folder / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                if (j + 1) % 10 == 0:
                    print(
                        json.dumps(
                            {
                                "context_variant": tag,
                                "wm_round": j + 1,
                                "seconds": elapsed,
                            }
                        ),
                        flush=True,
                    )
            assert t.actor_updates == 0
            for n in ("actor", "critic", "obs_norm", "critic_norm"):
                assert all(
                    torch.equal(v.detach().cpu(), fixed[n][k])
                    for k, v in getattr(t, n).state_dict().items()
                )
            t.save(folder / "checkpoint_final.pt")
            result = {
                "completed": True,
                "wm_input_dim": t.model.net[0].in_features,
                "wm_parameters": sum(p.numel() for p in t.model.parameters()),
                "replay_memory": t.replay.memory_report(),
                "model_updates": t.model_updates,
                "records": records,
                "cold_first_round_seconds": records[1]["wm_round_wall_seconds"],
                "mean_warm_round_seconds": (
                    sum(r["wm_round_wall_seconds"] for r in records[2:])
                    / (len(records) - 2)
                    if len(records) > 2
                    else None
                ),
                "policy_unchanged": True,
                "device_bytes_before_trial": memory_before,
            }
            report["results"][tag] = result
            (out / "context_ablation.json").write_text(json.dumps(report, indent=2))
            # Bound compiled methods can keep Trainer/Replay alive through Dynamo
            # caches. Release them before measuring the following representation.
            owner = weakref.ref(t)
            for name in (
                "_dynamics",
                "_post",
                "_fk",
                "_wm_training_step",
                "_v24_heads",
                "_v24_step",
                "_task_step",
                "_termination_fn",
                "_observations",
                "_transition",
                "_termination_before_fk",
                "_termination_after_fk",
            ):
                setattr(t, name, None)
            del t, task, scheduler
            torch._dynamo.reset()
            gc.collect()
            result["trainer_released"] = owner() is None
            if str(source.device).startswith("cuda"):
                torch.cuda.empty_cache()
                result["device_bytes_after_cleanup"] = torch.cuda.memory_allocated()
                (out / "context_ablation.json").write_text(json.dumps(report, indent=2))
        report["completed"] = True
        (out / "context_ablation.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        source.prior.load_amp_state_dict(amp)
        source.replay = Replay(
            frozen["capacity"], "cpu", frozen.get("global_fields", ())
        )
        source.replay.load_state_dict(frozen)
