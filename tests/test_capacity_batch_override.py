import json
from types import SimpleNamespace as NS

import pytest
import torch
from reflex.world_model.efficiency_benchmark import capacity_check


@pytest.mark.parametrize("oom", [False, True])
def test_explicit_batch_override_runs_full_batch_and_restores(
    tmp_path, monkeypatch, oom
):
    cfg = NS(
        batch=2048,
        actor_batch=2048,
        actor_microbatch=2048,
        capacity_steps=2,
        horizon=32,
    )
    calls = []

    def model():
        assert (cfg.batch, cfg.actor_batch, cfg.actor_microbatch) == (4096,) * 3
        calls.append("wm")
        return {"finite": True}

    def actor():
        assert cfg.actor_microbatch == 4096
        assert cfg.horizon == 16
        calls.append("actor")
        if oom:
            raise torch.OutOfMemoryError("test OOM")
        return {"finite": True}

    t = NS(
        cfg=cfg,
        device="cpu",
        collect=lambda a: calls.append("collect"),
        replay=NS(memory_report=lambda: {}),
        fit_model=model,
        update_actor=actor,
    )
    for name in ("reset_peak_memory_stats",):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 123)
    if oom:
        with pytest.raises(torch.OutOfMemoryError):
            capacity_check(t, None, tmp_path, batch_override=4096, horizon_override=16)
    else:
        assert capacity_check(
            t, None, tmp_path, batch_override=4096, horizon_override=16
        )["passed"]
        assert calls.count("actor") == calls.count("wm") == 4
    result = json.loads((tmp_path / "capacity_check.json").read_text())
    assert result["effective_batches"] == dict(
        batch=4096, actor_batch=4096, actor_microbatch=4096
    )
    assert result["passed"] is not oom
    assert cfg.batch == cfg.actor_batch == cfg.actor_microbatch == 2048
    assert cfg.horizon == 32
    assert result["effective_horizon"] == 16
