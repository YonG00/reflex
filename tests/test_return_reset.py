import torch
from reflex.world_model.return_reset import ReturnResetSampler


def test_probability_and_tail_frames():
    s = ReturnResetSampler(103, 10000, 0.99, 1.0)
    ids = torch.arange(10000)
    torch.manual_seed(2)
    frames = s.sample(ids)
    assert frames.min() >= 0 and frames.max() < 103
    assert torch.equal(s.reset_bin, frames // 50)
    s.returns[:] = torch.tensor([0.0, 100.0, 100.0])
    frames = s.sample(ids)
    assert (frames < 50).float().mean() > 0.55


def test_return_update_termination_timeout_and_duplicate_bins():
    s = ReturnResetSampler(100, 3, 0.9, 1.0)
    s.reset_bin[:] = torch.tensor([0, 0, 1])
    s.update(
        torch.arange(3),
        torch.tensor([True, False, True]),
        torch.tensor([2, 2, 1]),
        torch.tensor([2.0, 2.0, 3.0]),
        torch.tensor([1.9, 1.9, 3.0]),
    )
    expected = 0.2 * ((1.9 + (1.9 + 0.9**2 / 0.1)) / 2)
    torch.testing.assert_close(s.returns, torch.tensor([expected, 0.6]))


def test_native_hook_preserves_reward_and_only_changes_frame_selection():
    from types import SimpleNamespace

    from reflex.world_model.return_reset import NativeReturnReset

    command = SimpleNamespace(
        motion=SimpleNamespace(time_step_total=103),
        time_steps=torch.zeros(3, dtype=torch.long),
        _adaptive_sampling=lambda ids: None,
    )
    original = torch.tensor([1.0, 2.0, 3.0])
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command),
        reward_manager=SimpleNamespace(compute=lambda dt: original),
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([True, False, False])
        ),
    )
    hook = NativeReturnReset(env, 0.9, 3.0)
    assert env.reward_manager.compute(0.02) is original
    command._adaptive_sampling(torch.tensor([0, 1]))
    torch.testing.assert_close(
        hook.sampler.returns[0], torch.tensor(0.2 * (1 + 20.0) / 2)
    )
    assert hook.lengths.tolist() == [0, 0, 1]
    assert ((command.time_steps >= 0) & (command.time_steps < 103)).all()
    saved = hook.state_dict()
    hook.sampler.returns.zero_()
    hook.load_state_dict(saved)
    torch.testing.assert_close(hook.sampler.returns, saved["returns"])
    # A reference-only rollover also closes its segment and gets new reset bins.
    env.termination_manager.terminated.zero_()
    env.reward_manager.compute(0.02)
    command._adaptive_sampling(torch.tensor([2]))
    assert hook.lengths[2] == 0


def test_checkpoint_invalid_past_is_transactional():
    from types import SimpleNamespace

    import pytest
    from reflex.world_model.return_reset import NativeReturnReset

    command = SimpleNamespace(
        motion=SimpleNamespace(time_step_total=101),
        time_steps=torch.zeros(2, dtype=torch.long),
        _adaptive_sampling=lambda ids: None,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command),
        reward_manager=SimpleNamespace(compute=lambda dt: torch.zeros(2)),
        termination_manager=SimpleNamespace(
            terminated=torch.zeros(2, dtype=torch.bool)
        ),
    )
    hook = NativeReturnReset(env, 0.99, 1.0)
    state = hook.state_dict()
    state["returns"].fill_(10.0)
    state["past"].fill_(-1.0)
    with pytest.raises(ValueError, match="historical"):
        hook.load_state_dict(state)
    assert not hook.sampler.returns.any()


def test_sampler_checkpoint_through_trainer(tmp_path):
    import copy

    from reflex.world_model.return_reset import NativeReturnReset
    from test_repaired_joint import repaired

    def attach(t):
        hook = NativeReturnReset.__new__(NativeReturnReset)
        hook.sampler = ReturnResetSampler(101, 3, 0.99, 1.0)
        t.return_reset = hook
        t.cfg.return_reset_max_step_reward = 1.0

    t = repaired(tmp_path)
    attach(t)
    t.return_reset.sampler.returns[:] = torch.tensor([1.0, 2.0, 3.0])
    path = tmp_path / "reset.pt"
    t.save(path)
    resumed = repaired(tmp_path)
    resumed.cfg = copy.deepcopy(t.cfg)
    attach(resumed)
    resumed.load(path)
    torch.testing.assert_close(
        resumed.return_reset.sampler.returns,
        t.return_reset.sampler.returns,
        rtol=0,
        atol=0,
    )
