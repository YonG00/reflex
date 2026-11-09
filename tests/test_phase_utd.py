"""Phase-local UTD accounting and checkpoint resume checks."""

import copy

from test_repaired_joint import repaired


def test_phase_anchor_and_checkpoint_resume(tmp_path):
    t = repaired(tmp_path)
    t.cfg.schedule = "joint_reproduction"
    t.cfg.wm_only_rounds = 500
    t.cfg.joint_updates_per_env_step = 0.1
    t.wm_rounds_completed = 499
    t.iteration = 1286  # deliberately not aligned to new rate
    t.fit_model()
    assert t.joint_phase_start_iteration == 1287
    assert not t.actor_warmup_complete()
    expected = []
    for i in range(1287, 1317):
        t.iteration = i
        expected.append(t.updates_due())
    assert expected == [0] * 9 + [1] + [0] * 9 + [1] + [0] * 9 + [1]
    path = tmp_path / "phase.pt"
    t.save(path)
    other = repaired(tmp_path)
    other.cfg = copy.deepcopy(t.cfg)
    other.load(path)
    assert other.joint_phase_start_iteration == 1287
    assert other.updates_due() == t.updates_due()
