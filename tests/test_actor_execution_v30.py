"""Functional metadata, RNG and pathwise-gradient checks for opt-in execution."""

import types

import pytest
import torch
from reflex.world_model.events_v24.execution_v30 import configure
from reflex.world_model.events_v24.fast_view import build
from reflex.world_model.events_v24.runtime import EventTerm
from reflex.world_model.events_v24.writes import WritePlan
from reflex.world_model.joint import returns
from test_analytic_soft_termination import new_trainer
from test_compact_context import compact_trainer, fill
from test_v24_events import setup, toy_control


def perturb_all(env, env_ids=None):
    del env_ids
    box = env.scene["balloon"]
    value = box.data.root_vel_w + torch.rand_like(box.data.root_vel_w)
    value[:, :3] = value[:, :3] + env.scene["robot"].data.root_pos_w
    box.write_root_velocity_to_sim(value)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("pool_count", [0, 1, 2])
def test_pool_update_state_rng_gradient_and_input_immutability(
    tmp_path, shared, pool_count
):
    t, b, runtime, event = setup(tmp_path)
    if pool_count:
        runtime.terms = [
            EventTerm(str(i), perturb_all, {}, (0.5, 1.0), False, True)
            for i in range(pool_count)
        ]
        runtime.pool_terms = list(range(pool_count))
        runtime.pool_index = {i: i for i in range(pool_count)}
        runtime.codec.timer_count = pool_count
        runtime.codec.width = runtime.codec.control_width + pool_count
        values, _ = runtime.codec.decode(event)
        event = runtime.codec.encode(values, torch.full((4, pool_count), 0.9))
    pool = torch.full((4, pool_count, 4), 0.9)
    if pool_count:
        pool[0, :, 2] = 0
        pool[2, :, 2] = 0
        if shared:
            pool[:] = pool[:1].clone()
    before_pool, before_event = pool.clone(), event.clone()
    outputs = []
    for opt in (False, True):
        configure(runtime, compact_pools=opt)
        physical = b["state"].clone().requires_grad_()
        pp = pool.clone().requires_grad_()
        ee = event.clone().requires_grad_()
        torch.manual_seed(17)
        plan, after, pools = runtime.step(
            b,
            physical,
            t.bridge.fk(physical),
            ee,
            pp,
            torch.ones(4, dtype=torch.long),
            torch.ones(4, dtype=torch.long),
            shared_world=shared,
            isolated_rng=False,
        )
        loss = (
            plan.targets.square().sum()
            + after.square().sum() * 0.01
            + pools.square().sum() * 0.001
        )
        gradients = torch.autograd.grad(loss, (physical, ee, pp), allow_unused=True)
        outputs.append(
            (plan.targets, plan.mask, after, pools, gradients, torch.get_rng_state())
        )
        torch.testing.assert_close(pp, before_pool, atol=0, rtol=0)
        torch.testing.assert_close(ee, before_event, atol=0, rtol=0)
    for a, c in zip(outputs[0][:4], outputs[1][:4]):
        torch.testing.assert_close(a, c, atol=0, rtol=0)
    for a, c in zip(outputs[0][4], outputs[1][4]):
        assert (a is None) == (c is None)
        if a is not None:
            torch.testing.assert_close(a, c, atol=0, rtol=0)
    torch.testing.assert_close(outputs[0][5], outputs[1][5], atol=0, rtol=0)
    if pool_count and not shared:
        assert outputs[1][1][[0, 2], 78:84].all()
        assert not outputs[1][1][
            [1, 3]
        ].any()  # Foreign clock fires full callback only in its world.


@pytest.mark.parametrize(
    "write",
    [
        "box_pose",
        "robot_velocity",
        "robot_pose",
        "robot_joint",
        "equal_pose",
        "equal_const_pose",
        "masked_pose",
    ],
)
def test_pose_tracking_uses_issued_writes_and_preserves_gradient(tmp_path, write):
    t, b, runtime, event = setup(tmp_path)
    runtime._view = types.MethodType(build, runtime)
    configure(runtime, track_pose=True)
    physical = b["state"].clone().requires_grad_()
    plan = WritePlan(physical)
    values, _ = runtime.codec.decode(event)
    view = runtime._view(
        b,
        physical,
        t.bridge.fk(physical),
        values,
        torch.ones(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long),
        plan,
    )
    robot = view.scene["robot"]
    box = view.scene["balloon"]
    if write == "box_pose":
        box.write_root_pose_to_sim(
            torch.cat((box.data.root_pos_w, box.data.root_quat_w), -1)
        )
    elif write == "robot_velocity":
        robot.write_root_velocity_to_sim(robot.data.root_vel_w * 2)
    elif write == "robot_joint":
        robot.write_joint_state_to_sim(robot.data.joint_pos, robot.data.joint_vel)
    else:
        pose = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), -1)
        if write == "masked_pose":
            robot._reflex_event_write_mask = torch.zeros(4, 1, dtype=torch.bool)
        if write == "equal_const_pose":
            pose = pose.detach()
        robot.write_root_pose_to_sim(pose if write == "equal_pose" else pose * 1)
    assert runtime.robot_pose_write_issued == (
        write not in ("box_pose", "robot_velocity")
    )
    post = torch.where(plan.mask, plan.targets, physical)
    grad = torch.autograd.grad(post.sum(), physical)[0]
    assert torch.isfinite(grad).all()
    if write == "equal_const_pose":
        assert grad[:, :7].eq(0).all()


def robot_velocity_event(env, env_ids=None):
    toy_control(env, env_ids)
    ids = (env.episode_length_buf == 3).nonzero().flatten()
    if len(ids):
        robot = env.scene["robot"]
        robot.write_root_velocity_to_sim(robot.data.root_vel_w[ids] + 0.1, env_ids=ids)


def equal_constant_pose_event(env, env_ids=None):
    toy_control(env, env_ids)
    ids = (env.episode_length_buf == 3).nonzero().flatten()
    if len(ids):
        robot = env.scene["robot"]
        pose = torch.cat(
            (robot.data.root_pos_w[ids], robot.data.root_quat_w[ids]), -1
        ).detach()
        robot.write_root_pose_to_sim(pose, env_ids=ids)


def robot_joint_event(env, env_ids=None):
    toy_control(env, env_ids)
    ids = (env.episode_length_buf == 3).nonzero().flatten()
    if len(ids):
        robot = env.scene["robot"]
        robot.write_joint_state_to_sim(
            robot.data.joint_pos[ids].detach(), robot.data.joint_vel[ids], env_ids=ids
        )


@pytest.mark.parametrize(
    "callback,extra",
    [
        (toy_control, 0),
        (robot_velocity_event, 0),
        (equal_constant_pose_event, 1),
        (robot_joint_event, 1),
    ],
)
@pytest.mark.parametrize("truncate", [False, True])
def test_h32_fk_reuse_trajectory_gradient_and_optimizer_equivalence(
    tmp_path, callback, extra, truncate
):
    source, b = compact_trainer(tmp_path, mode="full")
    fill(source, b)
    source.cfg.pure_soft_survival = True
    source.event_scheduler.terms[0].function = callback
    trainers = []
    for opt in (False, True):
        torch.manual_seed(3030)
        t = new_trainer(
            source,
            minimal_event_view=True,
            compact_event_inputs=True,
            reuse_post_event_fk=opt,
            compact_event_pool_updates=opt,
            batch_imagination_amp=True,
            batch_imagination_critic=True,
        )
        t.cfg.horizon = 32
        t.event_scheduler.task = t.task
        t.model.load_state_dict(source.model.state_dict())
        t.replay.load_state_dict(source.replay.state_dict())
        trainers.append(t)
    start = {
        k: v[0].clone()
        for k, v in source.replay.data.items()
        if k not in source.replay.global_fields
    }
    start["event_pool"] = (
        source.replay.data["event_pool"][0].unsqueeze(0).expand(4, -1, -1).clone()
    )
    if truncate:
        start["episode_steps"].fill_(trainers[0].task.max_steps - 4)
        extra = 0
    outputs = []
    counts = []
    gradients = []
    for t in trainers:
        count = [0]
        fk = t._fk

        def counted(*args, _count=count, _fk=fk, **kwargs):
            _count[0] += 1
            return _fk(*args, **kwargs)

        t._fk = counted
        imagined = t.imagine({k: v.clone() for k, v in start.items()}, trace=True)
        counts.append(count[0])
        outputs.append(imagined)
        assert imagined["predicted_launch_probability"][1].bool().all()
        loss = -returns(imagined, t.critic, t.cfg.gamma).mean()
        t.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(torch.cat([p.grad.flatten() for p in t.actor.parameters()]))
        t.actor_opt.step()
    assert counts == [65, 33 + extra]
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)
    torch.testing.assert_close(gradients[0], gradients[1])
    for a, c in zip(trainers[0].actor.parameters(), trainers[1].actor.parameters()):
        torch.testing.assert_close(a, c)


@pytest.mark.parametrize("truncate", [False, True])
def test_sparse_actor_bootstrap_return_gradients_targets_and_adam(tmp_path, truncate):
    from reflex.world_model.joint import critic_targets

    source, _ = compact_trainer(tmp_path, mode="full")
    fill(source, _)
    source.cfg.pure_soft_survival = True
    trainers = []
    for sparse in (False, True):
        torch.manual_seed(3031)
        t = new_trainer(
            source,
            minimal_event_view=True,
            compact_event_inputs=True,
            reuse_post_event_fk=True,
            compact_event_pool_updates=True,
            batch_imagination_amp=True,
            batch_imagination_critic=True,
            sparse_actor_bootstrap=sparse,
        )
        t.cfg.horizon = 32
        t.event_scheduler.task = t.task
        t.model.load_state_dict(source.model.state_dict())
        t.replay.load_state_dict(source.replay.state_dict())
        trainers.append(t)
    start = {
        k: v[0].clone()
        for k, v in source.replay.data.items()
        if k not in source.replay.global_fields
    }
    start["event_pool"] = (
        source.replay.data["event_pool"][0].unsqueeze(0).expand(4, -1, -1).clone()
    )
    if truncate:
        start["episode_steps"] = torch.tensor(
            [trainers[0].task.max_steps - 4, 0, trainers[0].task.max_steps - 12, 0]
        )
    results = []
    for t in trainers:
        imagined = t.imagine({k: v.clone() for k, v in start.items()})
        result = returns(imagined, t.critic, t.cfg.gamma)
        grads = torch.autograd.grad(
            result.mean(), tuple(t.actor.parameters()), retain_graph=True
        )
        targets = critic_targets(t, imagined)
        t.actor_opt.zero_grad(set_to_none=True)
        (-result.mean()).backward()
        t.actor_opt.step()
        results.append((imagined, result, grads, targets))
    for key in results[0][0]:
        torch.testing.assert_close(
            results[0][0][key], results[1][0][key], atol=0, rtol=0
        )
    torch.testing.assert_close(results[0][1], results[1][1])
    for a, b in zip(results[0][2], results[1][2]):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(results[0][3], results[1][3], atol=0, rtol=0)
    for a, b in zip(trainers[0].actor.parameters(), trainers[1].actor.parameters()):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("include_fk", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_fused_actor_core_captures_forward_and_backward_without_changing_return(
    tmp_path, sparse, include_fk
):
    source, row = compact_trainer(tmp_path, mode="full")
    fill(source, row)
    source.cfg.pure_soft_survival = True
    trainers = []
    for fused in (False, True):
        torch.manual_seed(3032)
        t = new_trainer(
            source,
            minimal_event_view=True,
            compact_event_inputs=True,
            reuse_post_event_fk=True,
            compact_event_pool_updates=True,
            sparse_actor_bootstrap=sparse,
            batch_imagination_amp=True,
            batch_imagination_critic=True,
            compile_v24_step=True,
            v24_fusion_scope="packed_heads",
            compile_backend="aot_eager",
            fuse_actor_core=fused,
            actor_core_include_fk=include_fk,
        )
        t.cfg.horizon = 4
        t.event_scheduler.task = t.task
        t.model.load_state_dict(source.model.state_dict())
        trainers.append(t)
    start = {
        k: v[0].clone()
        for k, v in source.replay.data.items()
        if k not in source.replay.global_fields
    }
    start["event_pool"] = (
        source.replay.data["event_pool"][0].unsqueeze(0).expand(4, -1, -1).clone()
    )
    outputs = []
    for t in trainers:
        r = t.imagine({k: v.clone() for k, v in start.items()}, trace=True)
        value = returns(r, t.critic, t.cfg.gamma)
        grad = torch.autograd.grad(value.mean(), tuple(t.actor.parameters()))
        outputs.append((r, value, grad))
    for k in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][k], outputs[1][0][k], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=0, rtol=0)
    for a, b in zip(outputs[0][2], outputs[1][2]):
        torch.testing.assert_close(a, b)
    torch._dynamo.reset()


@pytest.mark.parametrize("masked", [False, True])
def test_h8_compiled_rollout_loss_gradients_and_adam(tmp_path, masked):
    import copy

    source, row = compact_trainer(tmp_path, mode="full")
    baseline = new_trainer(source)
    fill(baseline, row)
    baseline.cfg.model_window = 8
    baseline.cfg.model_steps = 1
    baseline.fit_normalizers()
    candidate = copy.deepcopy(baseline)
    candidate.cfg.compile_wm_rollout = True
    candidate._wm_rollout = torch.compile(
        candidate.v30_model_rollout, backend="aot_eager", fullgraph=True
    )
    sequence = baseline.sample(8)
    if masked:
        sequence["loss_mask"] = torch.ones_like(sequence["phase"], dtype=torch.bool)
        sequence["loss_mask"][3:, 0] = False
    # Both fits consume exactly the same padded window, with identical parameters.
    for trainer in (baseline, candidate):
        trainer.sample = lambda *args, **kwargs: {
            k: v.clone() for k, v in sequence.items()
        }
    a, b = baseline.fit_model(), candidate.fit_model()
    for key in a:
        if not key.startswith("time/"):
            assert b[key] == pytest.approx(a[key], rel=1e-6, abs=1e-7)
    for p, q in zip(baseline.model.parameters(), candidate.model.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
        torch.testing.assert_close(p, q)
    assert candidate.termination is None and candidate.termination_updates == 0
    torch._dynamo.reset()


def test_execution_switches_preserve_checkpoint_contract(tmp_path):
    source, row = compact_trainer(tmp_path, mode="full")
    baseline = new_trainer(source)
    fill(baseline, row)
    path = tmp_path / "v29.pt"
    baseline.save(path)
    candidate = new_trainer(
        baseline,
        minimal_event_view=True,
        compact_event_inputs=True,
        reuse_post_event_fk=True,
        compact_event_pool_updates=True,
        sparse_actor_bootstrap=True,
        fuse_actor_core=True,
        compile_wm_rollout=True,
        compile_v24_step=True,
        v24_fusion_scope="packed_heads",
        compile_backend="aot_eager",
    )
    # Existing compile flags are execution options too; loading must not reset learning progress.
    candidate.load(path)
    assert candidate.model_updates == baseline.model_updates
    assert candidate.actor_updates == baseline.actor_updates
    for name in ("actor", "critic", "model"):
        for key, value in getattr(baseline, name).state_dict().items():
            torch.testing.assert_close(
                getattr(candidate, name).state_dict()[key], value, atol=0, rtol=0
            )


def test_wm_benchmark_restores_parameters_adam_and_rng(tmp_path):
    import copy

    from reflex.world_model.wm_rollout_benchmark import check

    source, row = compact_trainer(tmp_path, mode="full")
    trainer = new_trainer(source, compile_backend="aot_eager")
    fill(trainer, row)
    trainer.cfg.model_window = 2
    trainer.cfg.model_steps = 1
    trainer.fit_normalizers()
    initial = copy.deepcopy(trainer.model.state_dict())
    rng = torch.get_rng_state()
    report = check(trainer, tmp_path, repeats=1)
    assert report["completed"]
    assert trainer.model_updates == 0 and trainer.wm_rounds_completed == 0
    assert trainer._wm_rollout is None
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, initial[key], atol=0, rtol=0)
    torch._dynamo.reset()


def test_actor_execution_probe_full_updates_and_restore(tmp_path):
    from reflex.world_model.imagination_batching import check

    source, row = compact_trainer(tmp_path, mode="full")
    trainer = new_trainer(
        source,
        minimal_event_view=True,
        compact_event_inputs=True,
        compile_v24_step=True,
        v24_fusion_scope="packed_heads",
        compile_backend="eager",
        batch_imagination_amp=True,
        batch_imagination_critic=True,
    )
    fill(trainer, row)
    trainer.cfg.horizon = 4
    trainer.cfg.actor_steps = 1
    trainer.cfg.gradient_audit_every = 0
    report = check(trainer, tmp_path, repeats=1, execution_variants=True)
    assert report["completed"] and report["numeric_passed"], report.get(
        "failure_details"
    )
    assert set(report["variants"]) == {
        "reference",
        "fk_reuse",
        "fk_pool",
        "sparse_bootstrap",
        "fused_core",
    }
    assert trainer.actor_updates == 0 and trainer._actor_core is None
    assert not trainer.cfg.sparse_actor_bootstrap
    assert not trainer.cfg.compact_event_pool_updates
    torch._dynamo.reset()


def test_eager_diagnostic_disables_and_restores_new_compile_handles(tmp_path):
    from reflex.world_model.diagnostics import eager_diagnostic

    source, _ = compact_trainer(tmp_path, mode="full")
    sentinel = object()
    source._actor_core = sentinel
    bootstrap = source._bootstrap_critic_observation
    with pytest.raises(RuntimeError, match="probe"), eager_diagnostic(source):
        assert source._actor_core is None
        assert source._bootstrap_critic_observation == source.critic_observation
        raise RuntimeError("probe")
    assert source._actor_core is sentinel
    assert source._bootstrap_critic_observation == bootstrap
