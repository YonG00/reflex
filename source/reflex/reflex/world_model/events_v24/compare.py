"""Bounded, paired GPU regression on one frozen real replay (no long training)."""

import copy
import dataclasses
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from ..timing import timestamp
from ..trainer import Trainer


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def run(adapter, source, output, collect_steps=128, rounds=3):
    if not source.cfg.deterministic_event_apply:
        raise ValueError(
            "Paired V24 comparison requires the deterministic source config"
        )
    if source.cfg.capacity_steps < collect_steps:
        raise ValueError("Replay capacity must retain the whole frozen collection")
    out = Path(output)
    for j in range(max(0, collect_steps - source.replay.count)):
        source.collect(adapter)
        if (j + 1) % 32 == 0:
            print(
                json.dumps({"paired_collection_frames": source.replay.count}),
                flush=True,
            )
    source.save(out / "paired_source.pt")
    frozen_count = source.replay.count
    coverage = {
        "transitions": source.real_transitions,
        "frames": frozen_count,
        "h8_windows": len(source.replay.indices(8)),
        "h32_windows": len(source.replay.indices(32)),
        "actual_launch_writes": int(
            (
                ~source.replay.ordered("launched").bool()
                & source.replay.ordered("after_launched").bool()
                & source.replay.ordered("event_mask")[..., 78:81].any(-1)
                & (source.replay.ordered("event_targets")[..., 78:81].norm(dim=-1) > 1)
            ).sum()
        ),
    }
    (out / "paired_coverage.json").write_text(json.dumps(coverage, indent=2))
    reset = source.replay.ordered("boundary").bool().clone()
    # Trainer.load inserts a synthetic cut on the final replay frame so that
    # fresh simulator state cannot be joined to the saved trajectory. It is
    # not a physical episode/reference reset and its targets remain unchanged.
    reset[-1] = False
    done = (
        source.replay.ordered("terminated").bool()
        | source.replay.ordered("timeout").bool()
    )
    if reset.any():
        torch.testing.assert_close(
            source.replay.ordered("next_state")[reset],
            source.replay.ordered("physical_next_state")[reset],
            rtol=0,
            atol=0,
        )
        assert not source.replay.ordered("event_mask")[reset].any()
    coverage.update(
        {
            "reset_boundaries": int(reset.sum()),
            "episode_ends": int(done.sum()),
            "reference_only_boundaries": int((reset & ~done).sum()),
            "reset_target_alignment_passed": True,
        }
    )
    (out / "paired_coverage.json").write_text(json.dumps(coverage, indent=2))
    results = {}
    for tag, deterministic in (("learned", False), ("deterministic", True)):
        seed(24024)
        cfg = dataclasses.replace(source.cfg, deterministic_event_apply=deterministic)
        task = copy.deepcopy(source.task)
        scheduler = copy.deepcopy(source.event_scheduler)
        scheduler.task = task
        t = Trainer(
            source.bridge,
            source.action_map,
            copy.deepcopy(source.prior),
            source.replay.data["context"].shape[-1],
            6,
            source.reward_names,
            source.contract,
            source.device,
            cfg,
            task=task,
            event_scheduler=scheduler,
        )
        for name in ("actor", "critic", "obs_norm", "critic_norm", "termination"):
            getattr(t, name).load_state_dict(getattr(source, name).state_dict())
        incompatible = t.model.load_state_dict(source.model.state_dict(), strict=False)
        if incompatible.unexpected_keys or any(
            not k.startswith("post.") for k in incompatible.missing_keys
        ):
            raise ValueError("Unexpected physical initialization mismatch")
        # Explicitly shared frozen real data; no environment interaction below.
        t.replay = copy.deepcopy(source.replay)
        t.normalized = source.normalized
        t.real_transitions = source.real_transitions
        folder = out / tag
        folder.mkdir(exist_ok=True)
        records = []
        seed(9100)
        scheduler.rng = (
            torch.Generator(device=scheduler.device).manual_seed(9100).get_state()
        )
        print(
            json.dumps({"paired_variant": tag, "stage": "initial_validation"}),
            flush=True,
        )
        records.append({"round": 0, **t.validate()})
        for j in range(rounds):
            seed(5100 + j)
            start = timestamp(t.device)
            print(
                json.dumps({"paired_variant": tag, "round": j + 1, "stage": "wm"}),
                flush=True,
            )
            metrics = t.fit_model()
            metrics["time/wm_ten_updates_seconds"] = timestamp(t.device) - start
            seed(7100 + j)
            scheduler.rng = (
                torch.Generator(device=scheduler.device)
                .manual_seed(7100 + j)
                .get_state()
            )
            print(
                json.dumps(
                    {
                        "paired_variant": tag,
                        "round": j + 1,
                        "stage": "actor",
                        "wm_seconds": metrics["time/wm_ten_updates_seconds"],
                    }
                ),
                flush=True,
            )
            with torch.autograd.set_detect_anomaly(
                os.environ.get("REFLEX_WM_ANOMALY") == "1"
            ):
                metrics.update(t.update_actor())
            seed(9100 + j)
            scheduler.rng = (
                torch.Generator(device=scheduler.device)
                .manual_seed(9100 + j)
                .get_state()
            )
            metrics.update(t.validate())
            t.iteration += 1
            records.append({"round": j + 1, **metrics})
            with (folder / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(records[-1]) + "\n")
            print(json.dumps({"paired_variant": tag, **records[-1]}), flush=True)
        t.save(folder / "checkpoint_final.pt")
        t.load(folder / "checkpoint_final.pt")
        if t.replay.count != frozen_count or t.actor_updates != rounds:
            raise AssertionError("Paired train-save-load counters mismatch")
        results[tag] = {
            "checkpoint_reload_passed": True,
            "actor_updates": t.actor_updates,
            "wm_updates": t.model_updates,
            "records": records,
        }
        del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "completed": True,
        "coverage": coverage,
        "rounds": rounds,
        "frozen_replay": True,
        "amp_train_calls_per_variant": rounds,
        "independent_amp_state": True,
        "results": results,
    }
    (out / "paired_summary.json").write_text(json.dumps(result, indent=2))
    return result
