"""Production-config paired Actor fusion study; strict failures stay visible."""

import os
from pathlib import Path

import torch

from .actor_execution_v32_check import check as paired_check
from .diagnostic_watch_v35 import atomic_json, cold_watch


def check(t, output, repeats=20, warmups=2):
    if not (t.cfg.compile_v24_step and t.cfg.v24_fusion_scope == "packed_heads"):
        raise ValueError("Require production packed-head configuration")
    if os.environ.get("TRITON_DEFAULT_FP_FUSION", "1") != "1":
        raise ValueError("Use production FMA setting, not the V35 no-FMA experiment")
    if t.cfg.fuse_actor_core or t._actor_core is not None:
        raise ValueError("Start from the unfused production reference")
    original = t._actor_core
    include_fk = t.cfg.actor_core_include_fk
    compiled = torch.compile(
        t.actor_physical_core,
        fullgraph=True,
        dynamic=False,
        backend=t.cfg.compile_backend,
    )
    path = Path(output) / "actor_fusion_v36.json"

    def apply(name):
        t._actor_core = None if name == "reference" else compiled

    def watch(label, result):
        result["execution_study"] = {
            "baseline": "configured production V32",
            "candidate": "Actor/observation/control/Physical WM; FK remains separate",
            "formal_training": False,
        }

        def progress(value):
            result["active_operation"] = value
            atomic_json(path, result)

        return cold_watch(
            t,
            label,
            progress,
            Path(output) / "cold_stacks.log",
            int(os.environ.get("V36_STACK_SECONDS", "60")),
        )

    try:
        return paired_check(
            t,
            output,
            repeats,
            warmups,
            variants={"reference": (False, include_fk), "fused_no_fk": (True, False)},
            flag_names=("fuse_actor_core", "actor_core_include_fk"),
            apply_variant=apply,
            cold_context=watch,
            report_name=path.name,
        )
    finally:
        t._actor_core = original
