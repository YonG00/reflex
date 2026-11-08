"""Paired event-block/carry trials, preserving the real H32 update and RNG."""

import hashlib
from pathlib import Path

import torch

from .actor_carry_v37 import carry
from .actor_execution_v32_check import check as paired_check
from .diagnostic_watch_v35 import atomic_json, cold_watch
from .events_v24.blocks_v37 import BlockWritePlan
from .events_v24.writes import WritePlan

VARIANTS = {
    "reference": (False, False),
    "block_writes": (True, False),
    "fused_carry": (False, True),
    "combined": (True, True),
}


def check(t, output, repeats=20, warmups=2):
    if not (
        t.cfg.condition_external_events
        and t.cfg.deterministic_event_apply
        and t.cfg.minimal_event_view
        and t.cfg.packed_pose_carry
    ):
        raise ValueError("Require the verified deterministic packed-pose baseline")
    if (
        t.cfg.block_event_writes
        or t.cfg.fused_actor_carry
        or t.cfg.contiguous_event_writes
    ):
        raise ValueError("Start V37 from baseline execution flags")
    scheduler = t.event_scheduler
    present = "event_plan_factory" in scheduler.__dict__
    original = getattr(scheduler, "event_plan_factory", WritePlan)
    old_carry = t._actor_carry
    compiled = torch.compile(
        carry, fullgraph=True, dynamic=False, backend=t.cfg.compile_backend
    )
    path = Path(output) / "actor_execution_v37.json"
    sources = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("actor_execution_v37_check.py", "actor_carry_v37.py")
    }
    sources["events_v24/blocks_v37.py"] = hashlib.sha256(
        Path(__file__).with_name("events_v24").joinpath("blocks_v37.py").read_bytes()
    ).hexdigest()

    def apply(name):
        blocks, fused = VARIANTS[name]
        scheduler.event_plan_factory = BlockWritePlan if blocks else original
        t._actor_carry = compiled if fused else old_carry

    def watch(label, result):
        result["source_sha256"].update(sources)
        result["research_only"] = True
        result["formal_training"] = False

        def progress(value):
            result["active_operation"] = value
            atomic_json(path, result)

        # No timed faulthandler thread: incompatible with this GPU image.
        return cold_watch(t, label, progress, Path(output) / "cold_stacks.log")

    try:
        return paired_check(
            t,
            output,
            repeats,
            warmups,
            variants=VARIANTS,
            flag_names=("block_event_writes", "fused_actor_carry"),
            apply_variant=apply,
            cold_context=watch,
            report_name=path.name,
        )
    finally:
        t._actor_carry = old_carry
        if present:
            scheduler.event_plan_factory = original
        else:
            del scheduler.event_plan_factory
