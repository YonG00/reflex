from types import SimpleNamespace as NS
import torch
from reflex.event_math import rethrow_mask
from reflex.world_model.events_v24.state import EventStateCodec
from reflex.world_model.events_v24.writes import WritePlan
from reflex.world_model.events_v24.runtime import EventTerm, NativeEventScheduler
from test_repaired_joint import repaired


def toy_control(env, env_ids=None):
    box = env.scene["balloon"]
    waiting = ~env._box_launched
    launch = waiting & (env.episode_length_buf >= env._box_launch_step)
    pose = torch.cat((env._box_p0, env._box_q0), -1)
    old = torch.cat((box.data.root_pos_w, box.data.root_quat_w), -1)
    box.write_root_pose_to_sim(torch.where(waiting[:, None], pose, old))
    velocity = torch.zeros_like(box.data.root_vel_w)
    velocity[:, :3] = env._box_v0 + env.scene["robot"].data.root_pos_w
    box.write_root_velocity_to_sim(
        torch.where(
            launch[:, None],
            velocity,
            torch.where(
                waiting[:, None], torch.zeros_like(velocity), box.data.root_vel_w
            ),
        )
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
    gradient = torch.autograd.grad(plan.condition(physical).sum(), physical)[0]
    assert gradient[:, :3].abs().sum() > 0


def test_noop_write_contains_no_observed_future_state():
    state = torch.randn(3, 84)
    state[:, 3:7] = torch.tensor([1.0, 0, 0, 0])
    state[:, 74:78] = torch.tensor([1.0, 0, 0, 0])
    plan = WritePlan(state)
    plan.record(state[:, 78:84], state[:, 78:84].clone(), None, list(range(78, 84)))
    assert not plan.mask.any() and plan.targets.eq(0).all()
    assert plan.condition(state).abs().max() < 1e-5


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


def test_v24_joint_train_save_load_and_h32_event_progression(tmp_path):
    import copy
    from reflex.world_model.trainer import Trainer
    from reflex.world_model.diagnostics import gradient_audit

    old, b, runtime, event = setup(tmp_path)
    old.fit_model()
    source = tmp_path / "physical.pt"
    old.save(source)
    cfg = copy.deepcopy(old.cfg)
    cfg.learn_external_events = True
    cfg.condition_external_events = True
    cfg.joint_predicted_termination = True
    cfg.variable_model_windows = True
    cfg.state_loss_reduction = "sum"
    cfg.horizon = 32
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
    metrics = t.fit_model()
    assert metrics["termination/probe_weighted_bce_wm_gradient_norm"] > 0
    assert "wm/launch_mode_bce" not in metrics
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
