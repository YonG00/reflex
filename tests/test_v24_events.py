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
