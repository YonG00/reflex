import copy
from types import SimpleNamespace

import pytest
import torch

from test_repaired_joint import repaired
from reflex.world_model.return_reset_migration import validate_return_reset_migration


def test_explicit_full_resume_adds_only_fresh_sampler(tmp_path):
    old = repaired(tmp_path)
    old.actor_updates = 805
    old.wm_rounds_completed = 1305
    path = tmp_path / 'old.pt'
    old.save(path)
    new = repaired(tmp_path)
    new.cfg = copy.deepcopy(old.cfg)
    new.cfg.return_reset_max_step_reward = 1.314
    # Real env constructs this hook before load; no old sampler state may load.
    new.return_reset = SimpleNamespace(load_state_dict=lambda s: pytest.fail('No old statistics'))
    with pytest.raises(ValueError, match='config/physical contract'):
        new.load(path, reinitialize_replay=True)
    new.load(path, reinitialize_replay=True, initialize_return_reset=True)
    assert new.actor_updates == 805 and new.actor_warmup_complete()
    assert new.replay.count == 0
    assert new.initialization['return_reset_migration']['sampler_statistics'] == 'fresh'
    for name in ('model', 'actor', 'critic', 'obs_norm', 'critic_norm'):
        a, b = getattr(old, name).state_dict(), getattr(new, name).state_dict()
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
    for name in ('model_opt', 'actor_opt', 'critic_opt'):
        assert getattr(old, name).state_dict() == getattr(new, name).state_dict()


def test_migration_rejects_discarding_existing_history():
    cfg = SimpleNamespace(return_reset_max_step_reward=1.314)
    state = {'config': {}, 'actor_updates': 805}
    with pytest.raises(ValueError, match='fresh simulator'):
        validate_return_reset_migration(state, cfg, False)
    state['return_reset'] = {'returns': [1]}
    with pytest.raises(ValueError, match='must be restored'):
        validate_return_reset_migration(state, cfg, True)
