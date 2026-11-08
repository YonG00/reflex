"""Block writes and carry fusion must preserve commands, RNG and BPTT."""

import copy

import pytest
import torch
from reflex.world_model.actor_carry_v37 import apply, carry
from reflex.world_model.actor_execution_v32 import packed_pose_carry
from reflex.world_model.actor_execution_v37_check import check
from reflex.world_model.events_v24.blocks_v37 import BlockWritePlan
from reflex.world_model.events_v24.writes import WritePlan
from test_actor_diagnostics_v31 import assert_tree_equal, fixture


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "slots", [list(range(71, 78)), list(range(78, 84)), [7, 9, 12], [7, 7]]
)
@pytest.mark.parametrize("rows", [None, [2, 0], [], [1, 1]])
def test_block_writes_overwrite_masks_values_and_gradients(slots, rows):
    state = torch.randn(3, 84, requires_grad=True)
    ids = None if rows is None else torch.tensor(rows, dtype=torch.long)
    n = 3 if rows is None else len(rows)
    values = [torch.randn(n, len(slots), requires_grad=True) for _ in range(2)]
    result = []
    for cls in (WritePlan, BlockWritePlan):
        plan = cls(state)
        for i, value in enumerate(values):
            plan.record(None, value, ids, slots, torch.arange(n)[:, None] % 2 == i)
            # Observe intermediate writes, as subsequent native callbacks do.
            before = torch.where(plan.mask.clone(), plan.targets, state)
            assert before.isfinite().all()
        post = torch.where(plan.mask, plan.targets, state)
        result.append(
            (
                post,
                plan.mask,
                torch.autograd.grad(post.square().sum(), (state, *values)),
            )
        )
    assert_tree_equal(*result)


def test_block_select_merge_and_retained_intermediate_gradient():
    x = torch.randn(4, 84, requires_grad=True)
    results = []
    for cls in (WritePlan, BlockWritePlan):
        plan = cls(x)
        plan.record(None, x[:, 78:84] * 2, None, list(range(78, 84)))
        earlier = torch.where(plan.mask.clone(), plan.targets, x)
        child = plan.select(torch.tensor([3, 1]))
        child.record(None, child.targets[:, 78:84] + 1, None, list(range(78, 84)))
        plan.merge(torch.tensor([3, 1]), child)
        # Exercise irregular fallback after root-block functional writes.
        plan.record(None, x[:2, [7, 9]], torch.tensor([1, 0]), [7, 9])
        output = torch.where(plan.mask, plan.targets, x)
        loss = output.square().sum() + earlier.square().sum()
        results.append((output, plan.mask, torch.autograd.grad(loss, x)[0]))
    assert_tree_equal(*results)


def test_compiled_carry_matches_packed_reference_and_vjp():
    valid = torch.tensor([True, False, True])
    args = [torch.randn(3, n, requires_grad=True) for n in (84, 84, 29, 29)]
    old = (
        {"root": torch.randn(3, 3, requires_grad=True)},
        {"root": torch.randn(3, 4, requires_grad=True)},
    )
    new = (
        {"root": torch.randn(3, 3, requires_grad=True)},
        {"root": torch.randn(3, 4, requires_grad=True)},
    )
    inputs = (
        *args,
        *old[0].values(),
        *old[1].values(),
        *new[0].values(),
        *new[1].values(),
    )
    poses = packed_pose_carry(old, new, valid)
    expected = (
        torch.where(valid[:, None], args[0], args[1]),
        torch.where(valid[:, None], args[2], args[3]),
        poses,
    )
    fn = torch.compile(carry, fullgraph=True, backend="aot_eager")
    actual = apply(fn, valid, *args, new, old)
    assert_tree_equal(expected, actual)

    def loss(output):
        state, action, poses = output
        return (
            state.square().sum()
            + action.square().sum()
            + sum(t.square().sum() for group in poses for t in group.values())
        )

    assert_tree_equal(
        torch.autograd.grad(loss(expected), inputs),
        torch.autograd.grad(loss(actual), inputs),
    )


def test_full_update_four_variants_restore_rng_adam_and_handles(tmp_path):
    t = fixture(tmp_path)
    t.cfg.packed_pose_carry = True
    t.cfg.compile_backend = "aot_eager"
    t.cfg.horizon = 32
    t.update_actor()  # Nonempty Adam state, rather than just a fresh optimizer.
    weights = copy.deepcopy(t.actor.state_dict())
    opt = copy.deepcopy(t.actor_opt.state_dict())
    rng = torch.get_rng_state().clone()
    events = t.event_scheduler.rng.clone()
    result = check(t, tmp_path / "trial", repeats=1, warmups=1)
    assert result["completed"] and result["numeric_passed"], result
    assert result["performance_usable"]
    assert set(result["variants"]) == {
        "reference",
        "block_writes",
        "fused_carry",
        "combined",
    }
    assert t._actor_carry is None
    assert not hasattr(t.event_scheduler, "event_plan_factory")
    assert not t.cfg.block_event_writes and not t.cfg.fused_actor_carry
    assert_tree_equal(t.actor.state_dict(), weights)
    assert_tree_equal(t.actor_opt.state_dict(), opt)
    assert_tree_equal(torch.get_rng_state(), rng)
    assert_tree_equal(t.event_scheduler.rng, events)


def rethrow_and_push(env, env_ids=None):
    from test_actor_execution_v30 import robot_velocity_event

    ids = (env.episode_length_buf == 6).nonzero().flatten()
    if len(ids):
        env._box_launched[ids] = False
        env._box_launch_step[ids] = env.episode_length_buf[ids] + 2
        env._box_v0[ids] = (
            torch.rand(len(ids), 3) + env.scene["robot"].data.root_pos_w[ids]
        )
    robot_velocity_event(env, env_ids)


def test_scheduler_h32_launch_rethrow_push_rng_and_continuous_gradient(tmp_path):
    import types

    from reflex.world_model.events_v24.execution_v30 import configure
    from reflex.world_model.events_v24.fast_view import build
    from test_v24_events import setup

    t, b, scheduler, initial = setup(tmp_path)
    scheduler._view = types.MethodType(build, scheduler)
    scheduler.terms[0].function = rethrow_and_push
    configure(
        scheduler, track_pose=True, compact_pools=True, collapse_full_subsets=True
    )
    outputs = []
    for cls in (WritePlan, BlockWritePlan):
        scheduler.event_plan_factory = cls
        scheduler.rng = torch.Generator().manual_seed(24).get_state()
        state = initial.clone()
        leaf = b["state"].clone().requires_grad_()
        physical = leaf
        pools = torch.empty(len(physical), 0, len(physical))
        records, modes = [], []
        for index in range(32):
            episode = torch.full((len(physical),), index + 1)
            plan, state, pools = scheduler.step(
                b, physical, t.bridge.fk(physical), state, pools, episode, episode
            )
            physical = torch.where(plan.mask, plan.targets, physical)
            records.append((physical, plan.mask.clone(), state.clone()))
            modes.append(scheduler.codec.mode(state).clone())
            if index == 1:
                launch_grad = torch.autograd.grad(
                    physical[:, 78:81].sum(), leaf, retain_graph=True
                )[0]
        assert not modes[0].any() and modes[1].all()
        assert not modes[5].any() and modes[7].all()
        assert launch_grad[:, :3].eq(1).all()
        gradient = torch.autograd.grad(
            physical.square().sum() + state.sum() * 0.01, leaf
        )[0]
        assert gradient.isfinite().all()
        outputs.append((records, scheduler.rng.clone(), launch_grad, gradient))
    assert_tree_equal(*outputs)


def test_checkpoint_old_execution_flags_restore_full_optimizer(tmp_path):
    from test_analytic_soft_termination import new_trainer

    source = fixture(tmp_path)
    source.cfg.compile_backend = "aot_eager"
    source.update_actor()
    path = tmp_path / "previous.pt"
    source.save(path)
    saved = torch.load(path, weights_only=False)
    saved["config"].pop("block_event_writes")
    saved["config"].pop("fused_actor_carry")
    torch.save(saved, path)
    t = new_trainer(
        source, block_event_writes=True, fused_actor_carry=True, packed_pose_carry=True
    )
    t.load(path)
    assert t.event_scheduler.event_plan_factory is BlockWritePlan
    assert t._actor_carry is not None
    assert_tree_equal(t.actor.state_dict(), source.actor.state_dict())
    assert_tree_equal(t.actor_opt.state_dict(), source.actor_opt.state_dict())


def test_failed_trial_restores_existing_factory_flags_and_rng(tmp_path, monkeypatch):
    t = fixture(tmp_path)
    t.cfg.packed_pose_carry = True
    t.cfg.compile_backend = "aot_eager"
    t.event_scheduler.event_plan_factory = WritePlan
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(
        t, "update_actor", lambda: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError, match="stop"):
        check(t, tmp_path / "failed", repeats=1, warmups=1)
    assert t.event_scheduler.event_plan_factory is WritePlan
    assert not t.cfg.block_event_writes and not t.cfg.fused_actor_carry
    assert t._actor_carry is None
    assert_tree_equal(torch.get_rng_state(), rng)


def test_equal_constant_write_keeps_mask_and_removes_physical_derivative():
    state = torch.randn(3, 84, requires_grad=True)
    plan = BlockWritePlan(state)
    plan.record(None, state[:, 78:84].detach(), None, list(range(78, 84)))
    assert plan.mask[:, 78:84].all()
    post = torch.where(plan.mask, plan.targets, state)
    gradient = torch.autograd.grad(post.sum(), state)[0]
    assert gradient[:, 78:84].eq(0).all()
    assert gradient[:, :78].eq(1).all()


def test_actual_native_reset_function_with_block_recorder(tmp_path, monkeypatch):
    import test_v24_events as native_test

    outputs = []
    for index, cls in enumerate((WritePlan, BlockWritePlan)):
        recorded = []

        def factory(physical, cls=cls, recorded=recorded):
            plan = cls(physical)
            recorded.append(plan)
            return plan

        monkeypatch.setattr(native_test, "WritePlan", factory)
        torch.manual_seed(311)
        directory = tmp_path / str(index)
        directory.mkdir()
        # Executes the actual reset_box_throw AST, with deterministic sampler
        # stub, and checks gradients/metadata within the existing native test.
        native_test.test_native_rethrow_preserves_saved_masks_scalars_and_robot_gradient(
            directory
        )
        assert len(recorded) == 1
        outputs.append(
            (recorded[0].targets, recorded[0].mask, torch.get_rng_state().clone())
        )
    assert_tree_equal(*outputs)
