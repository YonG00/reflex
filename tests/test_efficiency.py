"""CPU equivalence and resume checks; no CUDA speed or PhysX claims."""

import copy

import pytest
import torch
from reflex.world_model.replay import Replay
from test_repaired_joint import repaired


def make_replay(cache, steps=14):
    replay = Replay(11, cache_indices=cache)
    g = torch.Generator().manual_seed(41)
    for i in range(steps):
        replay.add(
            {
                "valid": torch.rand(10, generator=g) > 0.1,
                "start_valid": torch.rand(10, generator=g) > 0.2,
                "boundary": torch.rand(10, generator=g) > 0.9,
                "terminated": torch.rand(10, generator=g) > 0.9,
                "timeout": torch.zeros(10, dtype=torch.bool),
                "phase": torch.full((10,), i),
                "reference_length": torch.full((10,), 100),
                "state": torch.randn(10, 84, generator=g),
            }
        )
    return replay


@pytest.mark.parametrize("steps", [6, 11, 14])
@pytest.mark.parametrize("validation", [False, True])
def test_cached_sampling_identical_rng_and_payload(steps, validation):
    fast, ref = make_replay(True, steps), make_replay(False, steps)
    for sample in (
        lambda r: r.sample(8, 2, validation=validation),
        lambda r: r.sample_variable_windows(8, 8, validation),
        lambda r: r.sample_starts(8, validation),
    ):
        for _ in range(2):
            torch.manual_seed(55)
            a = sample(fast)
            rng = torch.get_rng_state()
            torch.manual_seed(55)
            b = sample(ref)
            assert torch.equal(rng, torch.get_rng_state())
            for k in a:
                assert torch.equal(a[k], b[k]), k


def test_cache_invalidation_edits_add_load_and_memory():
    r = make_replay(True)
    old = r.start_indices()
    assert r.start_indices() is old
    r.data["start_valid"].zero_()
    assert not len(r.start_indices())
    r.load_state_dict(make_replay(False).state_dict())
    assert len(r.start_indices())
    r.add({k: v[0].clone() for k, v in r.data.items()})
    assert r.start_indices() is not old
    report = r.memory_report(1000)
    assert report["payload_bytes"] == sum(
        v[0].numel() * v.element_size() * 1000 for v in r.data.values()
    )


def test_wm_only_boundary_and_real_train_save_resume(tmp_path):
    t = repaired(tmp_path)
    t.cfg.schedule = "joint_reproduction"
    t.cfg.wm_only_rounds = 500
    actor_before = copy.deepcopy(t.actor.state_dict())
    critic_before = copy.deepcopy(t.critic.state_dict())
    t.iteration = 98
    t.wm_rounds_completed = 498
    t.fit_model()
    assert t.wm_rounds_completed == 499
    with pytest.raises(RuntimeError, match="WM-only"):
        t.update_actor()
    path = tmp_path / "warm.pt"
    t.save(path)
    resumed = repaired(tmp_path)
    resumed.cfg = copy.deepcopy(t.cfg)
    resumed.load(path)
    assert resumed.iteration == 98
    assert resumed.wm_rounds_completed == 499
    resumed.fit_model()
    assert resumed.wm_rounds_completed == 500
    assert not resumed.ready({})
    for net, old in ((resumed.actor, actor_before), (resumed.critic, critic_before)):
        assert all(torch.equal(v, old[k]) for k, v in net.state_dict().items())
    resumed.fit_model()
    assert resumed.wm_rounds_completed == 501
    assert resumed.ready({})
    resumed.update_actor()
    assert resumed.actor_updates == 1


@pytest.mark.parametrize("actor_updates,expected", [(0, False), (3, True)])
def test_legacy_checkpoint_warmup_migration(tmp_path, actor_updates, expected):
    t = repaired(tmp_path)
    t.actor_updates = actor_updates
    p = tmp_path / "legacy.pt"
    t.save(p)
    state = torch.load(p, weights_only=False)
    state.pop("wm_warmup")
    for k in (
        "wm_only_rounds",
        "wm_compile_mode",
        "console_log_every",
        "tensorboard_flush_seconds",
    ):
        state["config"].pop(k)
    torch.save(state, p)
    t.cfg.wm_only_rounds = 500
    t.load(p)
    assert t.actor_warmup_complete() == expected
    assert t.wm_warmup_inherited == expected


def test_async_sample_timing_has_no_explicit_fence(tmp_path, monkeypatch):
    t = repaired(tmp_path)
    t.cfg.synchronize_replay_timing = False

    def forbidden(_):
        raise AssertionError("Replay timing must not synchronize")

    monkeypatch.setattr("reflex.world_model.trainer.timestamp", forbidden)
    assert t.sample(2)["state"].shape[:2] == (2, 4)


def test_schema_projection_preserves_global_event_pool(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    r = Replay(2, global_fields=("pool",))
    r.add(
        {
            "valid": torch.ones(6, dtype=torch.bool),
            "state": torch.zeros(6, 84),
            "pool": torch.zeros(3, 8),
        }
    )
    p = tmp_path / "schema.pt"
    torch.save({"replay": r.state_dict()}, p)
    output = subprocess.check_output(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/local/benchmark_replay.py"
            ),
            "--checkpoint",
            str(p),
            "--schema-only",
            "--target-envs",
            "12",
            "--target-capacity",
            "1000",
        ]
    )
    result = json.loads(output)
    assert result["target_field_bytes_per_vector_step"]["pool"] == 96
    assert result["target_payload_gib"] == (12 * (1 + 84 * 4) + 96) * 1000 / 2**30
