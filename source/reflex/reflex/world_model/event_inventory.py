"""Read-only runtime inventory before implementing the V24 scheduler."""

import torch
from .task import primitive


def inspect(adapter, trainer):
    env = adapter.env

    def capture():
        result = {}
        manager = env.event_manager
        for mode, configs in manager._mode_term_cfgs.items():
            terms = []
            for i, (name, cfg) in enumerate(
                zip(manager._mode_term_names[mode], configs)
            ):
                row = {
                    "name": name,
                    "function": cfg.func.__module__ + "." + cfg.func.__name__,
                    "parameters": primitive(cfg.params),
                }
                if mode == "interval":
                    timer = manager._interval_term_time_left[i]
                    row.update(
                        interval_range_s=list(cfg.interval_range_s),
                        is_global_time=cfg.is_global_time,
                        timer_shape=list(timer.shape),
                        timer_min=float(timer.min()),
                        timer_max=float(timer.max()),
                    )
                terms.append(row)
            result[mode] = terms
        state = {}
        for name, value in vars(env).items():
            if name.startswith(("_box_", "_hold_", "_v36_", "_v38_", "_balloon_")):
                if torch.is_tensor(value):
                    state[name] = {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                elif isinstance(value, (bool, int, float, str)):
                    state[name] = value
                else:
                    state[name] = {"type": type(value).__name__}
        return {"events": result, "control_state": state}

    before = capture()
    b = adapter.snapshot()
    raw = torch.zeros_like(b["previous_action"])
    target, blend = trainer.targets(b, raw)
    adapter.step(raw, target, blend)
    return {"optimizer_updates": 0, "before": before, "after_one_step": capture()}
