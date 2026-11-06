from types import SimpleNamespace as NS

import pytest
import torch
from reflex.event_math import event_write, rethrow_mask
from reflex.world_model.events_v24.runtime import EventTerm, NativeEventScheduler
from reflex.world_model.events_v24.state import EventStateCodec
from reflex.world_model.events_v24.writes import WritePlan
from test_repaired_joint import repaired


def toy_control(env, env_ids=None):
    box = env.scene["balloon"]
    waiting = ~env._box_launched
    launch = waiting & (env.episode_length_buf >= env._box_launch_step)
    pose = torch.cat((env._box_p0, env._box_q0), -1)
    old = torch.cat((box.data.root_pos_w, box.data.root_quat_w), -1)
    event_write(
        box,
        "write_root_pose_to_sim",
        torch.where(waiting[:, None], pose, old),
        write_mask=waiting[:, None],
    )
    velocity = torch.zeros_like(box.data.root_vel_w)
    velocity[:, :3] = env._box_v0 + env.scene["robot"].data.root_pos_w
    event_write(
        box,
        "write_root_velocity_to_sim",
        torch.where(
            launch[:, None],
            velocity,
            torch.where(
                waiting[:, None], torch.zeros_like(velocity), box.data.root_vel_w
            ),
        ),
        write_mask=waiting[:, None],
    )
    env._box_launched = env._box_launched | launch


def setup(tmp_path):
    trainer = repaired(tmp_path)
    b = trainer.sample(starts=True, n=4)
    n = len(b["state"])
    b["event_origins"] = torch.zeros(n, 3)
    b["event_agent"] = torch.arange(n)
    env = NS(
        num_envs=n,
        device="cpu",
        _box_p0=b["state"][:, 71:74].clone(),
        _box_q0=b["state"][:, 74:78].clone(),
        _box_v0=torch.ones(n, 3),
        _box_launch_step=torch.full((n,), 2),
        _box_launched=torch.zeros(n, dtype=torch.bool),
    )
    term = EventTerm("toy", toy_control, {}, (0.0, 0.0), False, True)
    runtime = NativeEventScheduler.__new__(NativeEventScheduler)
    runtime.task = trainer.task
    runtime.terms = [term]
    runtime.codec = EventStateCodec(env, [term])
    runtime.static = {}
    runtime.device = "cpu"
    runtime.rng = torch.Generator().manual_seed(24).get_state()
    runtime.pool_terms, runtime.pool_index = [], {}
    runtime.source_count = n
    state = runtime.codec.capture(env, [torch.zeros(n)])
    return trainer, b, runtime, state


def test_clock_advances_to_launch_and_reaim_uses_predicted_robot(tmp_path):
    trainer, b, runtime, state = setup(tmp_path)
    physical = b["state"].clone().requires_grad_()
    poses = trainer.bridge.fk(physical)
    pools = torch.empty(len(physical), 0, len(physical))
    plan, state, pools = runtime.step(
        b,
        physical,
        poses,
        state,
        pools,
        torch.ones(len(physical), dtype=torch.long),
        torch.ones(len(physical), dtype=torch.long),
    )
    assert not runtime.codec.mode(state).any()
    plan, state, pools = runtime.step(
        b,
        physical,
        poses,
        state,
        pools,
        torch.full((len(physical),), 2),
        torch.full((len(physical),), 2),
    )
    assert runtime.codec.mode(state).all()
    assert plan.mask[:, 78:81].all()
    torch.testing.assert_close(
        plan.targets[:, 78:81], torch.ones(len(physical), 3) + physical[:, :3]
    )
    post = torch.where(plan.mask, plan.targets, physical)
    gradient = torch.autograd.grad(post[:, 78:81].sum(), physical)[0]
    torch.testing.assert_close(gradient[:, :3], torch.ones_like(gradient[:, :3]))


def test_equal_value_constant_write_remains_explicit_command():
    state = torch.randn(3, 84)
    state[:, 3:7] = torch.tensor([1.0, 0, 0, 0])
    state[:, 74:78] = torch.tensor([1.0, 0, 0, 0])
    plan = WritePlan(state)
    plan.record(state[:, 78:84], state[:, 78:84].clone(), None, list(range(78, 84)))
    assert plan.mask[:, 78:84].all()
    assert not plan.mask[:, :78].any()
    torch.testing.assert_close(plan.targets[:, 78:84], state[:, 78:84])
    assert plan.condition(state)[:, 84:].abs().max() < 1e-5


def test_rethrow_cadence_shared_scalar_and_per_agent_clocks():
    count = torch.tensor([30, 29, 50])
    for step in range(11):
        torch.testing.assert_close(
            rethrow_mask(step, count, 30),
            (count >= 30)
            if step % 5 == 0
            else torch.zeros_like(count, dtype=torch.bool),
        )
    torch.testing.assert_close(
        rethrow_mask(torch.tensor([5, 10, 11]), count, 30),
        torch.tensor([True, False, False]),
    )


def test_conditioned_model_does_not_replace_outputs_or_leak_post_commands():
    from reflex.world_model.models import ConditionedWorldModel
    from test_world_model import robot_state

    model = ConditionedWorldModel(2, hidden_layers=(16, 16, 16))
    state = robot_state(2)
    action, context, pre = torch.zeros(2, 29), torch.zeros(2, 2), torch.zeros(2, 4)
    physical = model.predict_physical(state, action, context, pre)
    targets = torch.zeros_like(state, requires_grad=True)
    with torch.no_grad():
        targets[:, 78] = 7
    mask = torch.zeros_like(state, dtype=torch.bool)
    mask[:, 78:81] = True
    post = model.predict_post(physical, targets, mask)
    # A zero learned decoder does not execute the requested 7 m/s write.
    torch.testing.assert_close(post, physical)
    assert torch.autograd.grad(physical.sum(), targets, allow_unused=True)[0] is None
    loss = (post[:, 78] - 7).square().mean()
    grad = torch.autograd.grad(loss, tuple(model.post.parameters()))
    assert sum(float(x.square().sum()) for x in grad) > 0


def test_global_timer_metadata_is_stored_once_and_sampled_per_source_frame():
    from reflex.world_model.replay import Replay

    r = Replay(4, global_fields=("event_pool",))
    for j in range(3):
        r.add(
            {
                "valid": torch.ones(6, dtype=torch.bool),
                "boundary": torch.zeros(6, dtype=torch.bool),
                "terminated": torch.zeros(6, dtype=torch.bool),
                "timeout": torch.zeros(6, dtype=torch.bool),
                "phase": torch.full((6,), j),
                "reference_length": torch.full((6,), 20),
                "event_pool": torch.full((1, 6), float(j)),
            }
        )
    assert r.data["event_pool"].shape == (4, 1, 6)
    b = r.sample_starts(20)
    torch.testing.assert_close(b["event_pool"][:, 0, 0], b["phase"].float())
    seq = r.sample(20, 2)
    torch.testing.assert_close(seq["event_pool"][:, :, 0, 0], seq["phase"].float())
    clone = Replay(4, global_fields=("event_pool",))
    clone.load_state_dict(r.state_dict())
    assert len(clone.start_indices()) == len(r.start_indices())


@pytest.mark.parametrize("deterministic", [False, True])
def test_v24_joint_train_save_load_and_h32_event_progression(tmp_path, deterministic):
    import copy

    from reflex.world_model.diagnostics import gradient_audit
    from reflex.world_model.trainer import Trainer

    old, b, runtime, event = setup(tmp_path)
    old.fit_model()
    source = tmp_path / "physical.pt"
    old.save(source)
    cfg = copy.deepcopy(old.cfg)
    cfg.learn_external_events = True
    cfg.condition_external_events = True
    cfg.deterministic_event_apply = deterministic
    cfg.minimal_event_view = deterministic
    cfg.compact_event_inputs = deterministic
    cfg.joint_predicted_termination = True
    cfg.variable_model_windows = True
    cfg.state_loss_reduction = "sum"
    cfg.horizon = 32
    cfg.model_window = 8
    t = Trainer(
        copy.deepcopy(old.bridge),
        copy.deepcopy(old.action_map),
        old.prior,
        98,
        6,
        old.reward_names,
        old.contract,
        cfg=cfg,
        task=copy.deepcopy(old.task),
        event_scheduler=runtime,
    )
    runtime.task = t.task
    t.warm_start_macro(source)
    s = b["state"].clone()
    pools = s.new_empty(len(s), 0, len(s))
    for j in range(40):
        bb = {
            **b,
            "state": s,
            "phase": torch.full((len(s),), j),
            "episode_steps": torch.full((len(s),), j),
            "common_step": torch.full((len(s),), j),
            "launched": runtime.codec.mode(event),
            "learn_external_events": True,
            "event_conditioned": True,
        }
        physical = s.clone()
        physical[:, 80] -= 0.1962
        physical[:, 71:74] += physical[:, 78:81] * 0.02
        poses = t.bridge.fk(physical)
        result = t.task.evaluate_step(
            bb,
            s,
            physical,
            b["action"],
            b["previous_action"],
            bb["phase"],
            b["task_history"],
            bb["episode_steps"],
            b["next_external_event_seconds"],
        )
        plan, next_event, pools = runtime.step(
            bb,
            physical,
            poses,
            event,
            pools,
            bb["episode_steps"] + 1,
            bb["common_step"] + 1,
        )
        post = torch.where(plan.mask, plan.targets, physical)
        tr = {k: v for k, v in bb.items() if torch.is_tensor(v)}
        tr.update(
            next_state=post,
            physical_next_state=physical,
            event_state=event,
            event_pool=pools[0],
            event_targets=plan.targets,
            event_mask=plan.mask,
            next_event_state=next_event,
            external_boundary=plan.mask.any(-1),
            after_launched=runtime.codec.mode(next_event),
            valid=torch.ones(len(s), dtype=torch.bool),
            boundary=result.terminated | result.timeout,
            terminated=result.terminated,
            timeout=result.timeout,
            reward_terms=result.rewards,
            memory=torch.zeros(len(s), 6),
            next_memory=torch.zeros(len(s), 6),
            termination_features=result.termination_features,
        )
        t.replay.add(tr)
        s, event = post, next_event
    training_rng = runtime.rng.clone()
    metrics = t.fit_model()
    torch.testing.assert_close(runtime.rng, training_rng)
    assert metrics["termination/probe_weighted_bce_wm_gradient_norm"] > 0
    assert "wm/launch_mode_bce" not in metrics
    if deterministic:
        assert metrics["wm/post_control_in_objective"] == 0
        assert metrics["wm/state_loss"] == pytest.approx(
            metrics["wm/physical_state_loss"]
        )
        assert metrics["wm/loss"] == pytest.approx(
            metrics["wm/physical_state_loss"]
            + cfg.term_loss_weight * metrics["termination/train_bce"],
            rel=1e-5,
        )
    assert t.model_updates == t.termination_updates == cfg.model_steps
    # Explicitly start all imagined trajectories before launch.
    start = {
        k: v[0].clone()
        for k, v in t.replay.data.items()
        if k not in t.replay.global_fields
    }
    start["event_pool"] = (
        t.replay.data["event_pool"][0].unsqueeze(0).expand(len(s), -1, -1).clone()
    )
    imagined = t.imagine(start)
    assert not imagined["predicted_launch_probability"][0].bool().any()
    assert imagined["predicted_launch_probability"][1].bool().all()
    if deterministic:
        from dataclasses import replace
        from reflex.world_model.joint import returns

        original_evaluate = t.task.evaluate_step

        def terminal_step(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            return replace(result, terminated=torch.ones_like(result.terminated))

        t.task.evaluate_step = terminal_step
        t.cfg.hard_imagination_boundaries = True
        stopped = t.imagine(start)
        assert stopped["valid"][0].all()
        assert not stopped["valid"][1:].any()
        assert stopped["done"][0].eq(1).all()
        # Terminal reward is counted once; all later reward and bootstrap vanish.
        actual = returns(stopped, lambda x: torch.ones(*x.shape[:-1], 1), t.cfg.gamma)
        torch.testing.assert_close(actual, stopped["reward"][0])
        changed = {**stopped, "reward": stopped["reward"].clone()}
        changed["reward"][1:] = 10000
        torch.testing.assert_close(
            returns(changed, lambda x: torch.ones(*x.shape[:-1], 1), t.cfg.gamma),
            actual,
        )
        t.cfg.hard_imagination_boundaries = False
        t.cfg.pure_soft_survival = True
        soft = t.imagine(start)
        assert soft["valid"][1].all()
        assert (soft["done"][0] < 1).all()
        assert soft["predicted_launch_probability"][1].bool().all()
        t.cfg.pure_soft_survival = False
        t.task.evaluate_step = original_evaluate
    before_rng = runtime.rng.clone()
    audit = gradient_audit(t)
    torch.testing.assert_close(runtime.rng, before_rng)
    assert audit["diagnostic/total_gradient_norm"] > 0
    updated = t.update_actor()
    assert updated["actor/gradient_norm"] > 0
    path = tmp_path / "v24.pt"
    t.save(path)
    t.load(path)
    assert t.actor_updates == 1
    if deterministic:
        # Explicit new-world resume retains all learned states but cannot reuse
        # environment-indexed replay. Other contracts remain strict.
        source_count = runtime.source_count
        runtime.source_count = source_count + 1
        with pytest.raises(ValueError, match="learning layout"):
            t.load(path)
        t.load(path, reinitialize_replay=True)
        assert t.actor_updates == 1
        assert t.replay.count == 0
        assert t.iteration == 0
        assert t.initialization["replay_reinitialization"]["optimizers_preserved"]
        assert t.actor_opt.state_dict()["state"]
        runtime.source_count = source_count
        t.load(path)
        t.cfg.pure_soft_survival = True
        with pytest.raises(ValueError):
            t.load(path)
        t.load(path, migrate_soft_survival=True)
        assert t.initialization["soft_survival_migration"]["optimizers_preserved"]
        t.cfg.pure_soft_survival = False
        t.load(path)
        t.cfg.hard_imagination_boundaries = True
        with pytest.raises(ValueError):
            t.load(path)
        t.load(path, migrate_imagination_boundaries=True)
        assert t.actor_updates == 1
        assert t.initialization["imagination_boundary_migration"][
            "optimizers_preserved"
        ]
        t.cfg.hard_imagination_boundaries = False
        t.load(path)
        checkpoint = torch.load(path, weights_only=False)
        schedule_trial = copy.deepcopy(checkpoint)
        schedule_trial["config"]["schedule"] = "joint_reproduction"
        schedule_path = tmp_path / "schedule_trial.pt"
        torch.save(schedule_trial, schedule_path)
        original_schedule, original_rate = t.cfg.schedule, t.cfg.updates_per_env_step
        t.cfg.schedule, t.cfg.updates_per_env_step = "joint_reproduction", 0.1
        with pytest.raises(ValueError):
            t.load(schedule_path, reinitialize_replay=True)
        t.load(schedule_path, reinitialize_replay=True, migrate_update_rate=True)
        assert t.initialization["update_rate_migration"]["to"] == 0.1
        assert t.replay.count == 0 and t.actor_updates == 1
        t.cfg.schedule, t.cfg.updates_per_env_step = original_schedule, original_rate
        t.load(path)
        old_execution = copy.deepcopy(checkpoint)
        old_execution["learning_contract"]["event_scheduler"]["source_sha256"][
            "runtime.py"
        ] = "95e25b24ebf61455b80b43f5b1287dc39e6a1f81fa2d0230f2f7b1d0ead249fc"
        old_execution_path = tmp_path / "old_event_execution.pt"
        torch.save(old_execution, old_execution_path)
        with pytest.raises(ValueError, match="learning layout"):
            t.load(old_execution_path)
        t.load(old_execution_path, migrate_event_execution=True)
        assert t.initialization["event_execution_migration"]["optimizers_preserved"]
        t.load(path)
        legacy_execution = copy.deepcopy(checkpoint)
        legacy_execution["config"].pop("minimal_event_view")
        legacy_path = tmp_path / "legacy_execution.pt"
        torch.save(legacy_execution, legacy_path)
        t.load(legacy_path)
        assert t.cfg.minimal_event_view and t.actor_updates == 1
        checkpoint["config"]["deterministic_event_apply"] = False
        incompatible = tmp_path / "incompatible.pt"
        torch.save(checkpoint, incompatible)
        with pytest.raises(ValueError):
            t.load(incompatible)
        from reflex.world_model.events_v24.compare import run

        paired_dir = tmp_path / "paired"
        paired_dir.mkdir()
        paired = run(None, t, paired_dir, collect_steps=0, rounds=1)
        assert paired["completed"] and paired["frozen_replay"]
        assert paired["results"]["learned"]["checkpoint_reload_passed"]
        assert paired["results"]["deterministic"]["checkpoint_reload_passed"]
        from reflex.world_model.events_v24.actor_profile import update_equivalence

        actor_before = copy.deepcopy(t.actor.state_dict())
        update_pair = update_equivalence(t, paired_dir)
        assert update_pair["passed"] and update_pair["initial_state_restored"]
        assert t.actor_updates == 1
        from reflex.world_model.events_v24.fused_step import check

        torch._dynamo.reset()  # Different test backends/shapes must not share an 8-entry cache.
        old_step, old_obs = t._v24_step, t._observations
        t._v24_step = torch.compile(
            t.v24_physical_step, fullgraph=True, backend="aot_eager"
        )
        t._observations = torch.compile(
            t.imagined_observations, fullgraph=True, backend="aot_eager"
        )
        try:
            fused = check(t, paired_dir, repeats=1)
            assert fused["equivalence_passed"] and fused["optimizer_updates"] == 0
        finally:
            t._v24_step, t._observations = old_step, old_obs
        torch._dynamo.reset()
        original_heads, original_scope = t._v24_heads, t.cfg.v24_fusion_scope
        t.cfg.v24_fusion_scope = "packed_heads"
        t._v24_heads = torch.compile(
            t.v24_packed_reward_termination,
            fullgraph=True,
            backend="aot_eager",
            dynamic=False,
        )
        t._v24_step = t.v24_partitioned_step
        t._observations = torch.compile(
            t.imagined_observations, fullgraph=True, backend="aot_eager", dynamic=False
        )
        try:
            packed = check(t, paired_dir, repeats=1)
            assert packed["equivalence_passed"]
            torch._dynamo.reset()
            fused_update = update_equivalence(t, paired_dir, fusion=True)
            assert fused_update["passed"] and fused_update["initial_state_restored"]
        finally:
            t._v24_heads, t.cfg.v24_fusion_scope = original_heads, original_scope
            t._v24_step, t._observations = old_step, old_obs
        for key, value in t.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)


def test_batch_coupled_timer_uses_foreign_due_clock_without_changing_native_law(
    tmp_path,
):
    trainer, b, runtime, state = setup(tmp_path)

    def perturb(env, env_ids):
        del env_ids
        old = env.scene["balloon"].data.root_vel_w
        env.scene["balloon"].write_root_velocity_to_sim(old + torch.rand_like(old))

    runtime.terms = [EventTerm("perturb", perturb, {}, (0.5, 1.0), False, True)]
    runtime.pool_terms, runtime.pool_index = [0], {0: 0}
    physical = b["state"]
    poses = trainer.bridge.fk(physical)
    timer = torch.tensor([0.9, 0.9, 0.0, 0.9])
    state[:, -1] = timer
    pool = timer[None, None, :].expand(4, 1, 4).clone()
    values, _ = runtime.codec.decode(state)
    expected = WritePlan(physical)
    view = runtime._view(
        b,
        physical,
        poses,
        values,
        torch.ones(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
        expected,
    )
    torch.manual_seed(24)
    remaining = timer - 0.02
    due = (remaining < 1e-6).nonzero().flatten()
    remaining[due] = torch.rand(len(due)) * 0.5 + 0.5
    perturb(view, due)
    torch.manual_seed(24)
    actual, next_state, _ = runtime.step(
        b,
        physical,
        poses,
        state,
        pool,
        torch.ones(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
        shared_world=True,
        isolated_rng=False,
    )
    torch.testing.assert_close(actual.targets, expected.targets)
    assert actual.mask[:, 78:84].all()  # Native callback ignores env_ids.
    torch.testing.assert_close(next_state[:, -1], remaining)
    # Independent imagined source worlds keep the original world-wide trigger.
    pool[1] = 0.9
    pool[3] = 0.9
    actual, _, _ = runtime.step(
        b,
        physical,
        poses,
        state,
        pool,
        torch.ones(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
    )
    assert actual.mask[[0, 2], 78:84].all()
    assert not actual.mask[[1, 3]].any()


def test_deterministic_apply_preserves_unwritten_state_and_target_gradients():
    from reflex.world_model.models import DeterministicEventWorldModel

    model = DeterministicEventWorldModel(2, hidden_layers=(16, 16, 16))
    assert not any(k.startswith("post.") for k in model.state_dict())
    physical = torch.randn(3, 84, requires_grad=True)
    targets = physical * 2
    mask = torch.zeros_like(physical, dtype=torch.bool)
    mask[:, 78:81] = True
    post = model.predict_post(physical, targets, mask)
    assert torch.equal(post[~mask], physical[~mask])
    assert torch.equal(post[mask], targets[mask])
    grad = torch.autograd.grad(post.sum(), physical)[0]
    assert torch.equal(grad, torch.where(mask, 2.0, 1.0))


def test_deterministic_migration_preserves_v24_physical_features():
    from reflex.world_model.models import (
        ConditionedWorldModel,
        DeterministicEventWorldModel,
    )

    old = ConditionedWorldModel(2, hidden_layers=(16, 16, 16))
    new = DeterministicEventWorldModel(2, hidden_layers=(16, 16, 16))
    with torch.no_grad():
        old.inputs.mean.copy_(torch.randn_like(old.inputs.mean))
        old.inputs.std.fill_(2)
    new.warm_start_physical(old.state_dict())
    for key, value in new.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[key])
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict())


def test_native_write_plan_frames_subsets_and_last_write_wins():
    from reflex.world_model.events_v24.writes import record_native
    from test_world_model import robot_state

    physical = robot_state(4)
    origins = torch.tensor(
        [[10.0, 2.0, 0.0], [-10.0, 4.0, 0.0], [0.0, 9.0, 1.0], [3.0, 2.0, 0.0]]
    )
    plan = WritePlan(physical)
    ids = torch.tensor([1, 3])
    local = physical[ids, 71:78].clone()
    local[:, :3] += 0.25
    world = local.clone()
    world[:, :3] += origins[ids]
    record_native(
        plan,
        physical,
        origins,
        False,
        "write_root_pose_to_sim",
        (world,),
        {"env_ids": ids},
    )
    velocity = torch.arange(12).reshape(2, 6).float()
    record_native(
        plan,
        physical,
        origins,
        False,
        "write_root_velocity_to_sim",
        (velocity,),
        {"env_ids": ids},
    )
    joints = torch.tensor([0, 5, 28])
    record_native(
        plan,
        physical,
        origins,
        True,
        "write_joint_state_to_sim",
        (torch.ones(2, 3), torch.full((2, 3), 2.0)),
        {"joint_ids": joints, "env_ids": ids},
    )
    expected = physical.clone()
    expected[ids, 71:78] = local
    expected[ids, 78:84] = velocity
    expected[ids[:, None], (joints + 7)[None, :]] = 1
    expected[ids[:, None], (joints + 42)[None, :]] = 2
    torch.testing.assert_close(torch.where(plan.mask, plan.targets, physical), expected)
    # Explicit equal-value writes and sub-epsilon writes must still overwrite.
    tiny = physical[ids, 78:84].clone()
    tiny[:, 0] += 1e-8
    record_native(
        plan,
        physical,
        origins,
        False,
        "write_root_velocity_to_sim",
        (tiny,),
        {"env_ids": ids},
    )
    assert plan.mask[ids, 78:84].all()
    assert torch.equal(plan.targets[ids, 78:84], tiny)


def test_passthrough_provenance_does_not_teacher_force_equal_constant_writes(tmp_path):
    trainer, b, runtime, event = setup(tmp_path)
    values, timers = runtime.codec.decode(event)
    values["_box_launched"][:] = True
    event = runtime.codec.encode(values, timers)
    physical = b["state"].clone().requires_grad_()
    plan, _, _ = runtime.step(
        b,
        physical,
        trainer.bridge.fk(physical),
        event,
        physical.new_empty(len(physical), 0, len(physical)),
        torch.full((len(physical),), 3),
        torch.full((len(physical),), 3),
    )
    assert not plan.mask.any()  # Native full-buffer writes are explicit identity.
    torch.testing.assert_close(torch.where(plan.mask, plan.targets, physical), physical)
    assert torch.equal(
        torch.autograd.grad(
            torch.where(plan.mask, plan.targets, physical).sum(), physical
        )[0],
        torch.ones_like(physical),
    )
    # Equal-valued constant interventions remain writes, unlike identity fields.
    explicit = WritePlan(physical)
    explicit.record(
        physical[:, 78:84],
        physical[:, 78:84].detach(),
        None,
        list(range(78, 84)),
        torch.ones(len(physical), 1, dtype=torch.bool),
    )
    assert explicit.mask[:, 78:84].all()
    post = torch.where(explicit.mask, explicit.targets, physical)
    assert torch.autograd.grad(post[:, 78:84].sum(), physical)[0][:, 78:84].eq(0).all()


def test_event_write_preserves_native_arguments_rng_and_restores_metadata():
    calls = []
    target = torch.randn(2, 6)
    ids = torch.tensor([0, 2])
    mask = torch.tensor([[True], [False]])
    asset = NS(_reflex_event_write_mask="outer")

    def native(value, env_ids=None):
        assert value is target and env_ids is ids
        assert asset._reflex_event_write_mask is mask
        calls.append(1)
        return "native-result"

    asset.write_root_velocity_to_sim = native
    rng = torch.get_rng_state().clone()
    assert (
        event_write(
            asset, "write_root_velocity_to_sim", target, env_ids=ids, write_mask=mask
        )
        == "native-result"
    )
    assert calls == [1]
    assert asset._reflex_event_write_mask == "outer"
    assert torch.equal(torch.get_rng_state(), rng)


def test_native_rethrow_preserves_saved_masks_scalars_and_robot_gradient(tmp_path):
    """Execute the actual native reset body without importing Isaac on CPU."""
    import ast
    import os
    from pathlib import Path

    trainer, b, runtime, packed = setup(tmp_path)
    physical = b["state"].clone().requires_grad_()
    values, _ = runtime.codec.decode(packed)
    plan = WritePlan(physical)
    env = runtime._view(
        b,
        physical,
        trainer.bridge.fk(physical),
        values,
        torch.ones(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
        plan,
    )
    env._box_lib_mask = torch.ones(4, dtype=torch.bool)
    env._box_tmult = torch.ones(4)
    saved = torch.where(
        env._box_lib_mask[:, None], env._box_v0, env.scene["robot"].data.root_pos_w
    )
    saved = saved + env.scene["robot"].data.root_pos_w * env._box_tmult[:, None]

    def samples(n, anchor, fwd, device, step_dt):
        del fwd, step_dt
        return {
            "p0": anchor + 1,
            "q0": torch.tensor([1.0, 0, 0, 0]).expand(n, 4),
            "v0": anchor + 2,
            "omega": torch.zeros(n, 3),
            "hover_steps": torch.full((n,), 5),
            "aim_err": torch.zeros(n),
            "catch_h": torch.ones(n),
            "t_mult": torch.full((n,), 2.0),
            "jitter": torch.ones(n),
            "lib_mask": torch.zeros(n, dtype=torch.bool),
        }

    path = Path(__file__).parents[1] / "source/reflex/reflex/envs/throw.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "reset_box_throw"
    )
    namespace = {
        "torch": torch,
        "os": os,
        "SceneEntityCfg": lambda name: NS(name=name),
        "mix_models": lambda _: None,
        "BOX_MODELS": [],
        "sample_throws": samples,
        "THROW_LIB_ALIGN": False,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    namespace["reset_box_throw"](env, torch.arange(4))
    gradient = torch.autograd.grad(saved.sum() + plan.targets.sum(), physical)[0]
    assert torch.isfinite(gradient).all() and gradient[:, :3].abs().sum() > 0
    assert not env._box_lib_mask.any()
    assert env._box_tmult.eq(2).all()


def test_minimal_event_view_matches_full_view_commands_and_gradients(tmp_path):
    import types

    from reflex.world_model.events_v24.fast_view import build

    trainer, b, runtime, event = setup(tmp_path)
    original_view = runtime._view
    rng = runtime.rng.clone()
    outputs = []
    original_body = trainer.task._body_data
    for minimal in (False, True):
        physical = b["state"].clone().requires_grad_()
        poses = trainer.bridge.fk(physical)
        runtime.rng = rng.clone()
        if minimal:
            runtime._view = types.MethodType(build, runtime)

            def forbidden(*args, **kwargs):
                raise AssertionError(
                    "Native event facade must not construct body velocity FK"
                )

            trainer.task._body_data = forbidden
        try:
            plan, after, pool = runtime.step(
                b,
                physical,
                poses,
                event,
                physical.new_empty(4, 0, 4),
                torch.full((4,), 2),
                torch.full((4,), 2),
            )
            gradient = torch.autograd.grad(plan.targets[:, 78:81].sum(), physical)[0]
            outputs.append((plan.targets, plan.mask, after, pool, gradient))
        finally:
            runtime._view = original_view
            trainer.task._body_data = original_body
    for x, y in zip(*outputs):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_critic_anchor_self_rotation_is_exact_identity_with_zero_gradient(tmp_path):
    t, b, _, _ = setup(tmp_path)
    state = b["state"].clone().requires_grad_()
    poses = t.bridge.fk(state)
    raw = t.observation(b, state, fk_result=poses)
    critic = t.privileged_bridge(state, fk_result=poses, actor_result=raw)
    anchor_index = t.privileged_bridge.tracked_names.index(t.bridge.anchor_name)
    start = 67 + 42 + 6 * anchor_index
    identity = state.new_tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expand(len(state), 6)
    assert torch.equal(critic[:, start : start + 6], identity)
    gradient = torch.autograd.grad(critic[:, start : start + 6].sum(), state)[0]
    assert gradient.eq(0).all()
