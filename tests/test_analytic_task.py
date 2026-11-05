"""CPU checks for state-derived velocities, deterministic history and truncation."""

from pathlib import Path
import xml.etree.ElementTree as ET
import torch
from reflex.world_model.physics import Kinematics, integrate
from reflex.world_model.task import AnalyticTask, Term, HISTORY_FIELDS

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/main_hands.urdf"


def state(n=2, dtype=torch.float32):
    s = torch.zeros(n, 84, dtype=dtype)
    s[:, 2] = 0.8
    s[:, 3] = s[:, 74] = 1
    s[:, 73] = 1.0
    return s


def joint_names():
    return [
        j.get("name")
        for j in ET.parse(URDF).getroot().findall("joint")
        if j.get("type") != "fixed"
    ]


def test_fk_world_link_velocities_match_finite_difference():
    torch.manual_seed(2)
    torch.set_num_threads(1)
    fk = Kinematics(URDF, joint_names()).double()
    s = state(dtype=torch.float64)
    s[:, 7:36] = torch.randn(2, 29, dtype=s.dtype) * 0.2
    s[:, 36:71] = torch.randn(2, 35, dtype=s.dtype) * 0.2
    com = torch.tensor([[0.0, 0.0, -0.07605]], dtype=s.dtype).expand(2, 3)
    p, q, v, w = fk.velocities(s, com)
    eps = 1e-6
    ns = integrate(s, torch.zeros(2, 82, dtype=s.dtype), eps, root_com=com)
    np, _ = fk(ns)
    for name in p:
        torch.testing.assert_close(
            (np[name] - p[name]) / eps, v[name], atol=2e-6, rtol=1e-5
        )


def make_task():
    fk = Kinematics(URDF, joint_names())
    s = state(1)
    p, q = fk(s)
    bodies = ["pelvis", "torso_link"]
    T = 16
    ref = {
        "q": torch.zeros(T, 29),
        "dq": torch.zeros(T, 29),
        "p": torch.stack([p[n][0] for n in bodies])[None].expand(T, 2, 3),
        "quat": torch.stack([q[n][0] for n in bodies])[None].expand(T, 2, 4),
        "v": torch.zeros(T, 2, 3),
        "w": torch.zeros(T, 2, 3),
        "arm_columns": torch.empty(0, dtype=torch.long),
        "arm_down": torch.empty(0),
    }

    def height(env):
        return (
            env.scene["robot"].data.root_pos_w[:, 2] + env.action_manager.action[:, 0]
        )

    def persistence(env):
        env._chest_v5_persist = torch.where(
            env.scene["balloon"].data.root_pos_w[:, 2] > 0.5,
            env._chest_v5_persist + 1,
            0,
        )
        return env._chest_v5_persist.float()

    def fall(env):
        return env.scene["robot"].data.root_pos_w[:, 2] < 0.3

    task = AnalyticTask(
        fk,
        bodies,
        joint_names(),
        bodies,
        "torso_link",
        ref,
        [
            Term("height", height, {}, 2.0),
            Term("persistent", persistence, {}, 0.5),
            Term("disabled", height, {}, 0.0),
        ],
        [Term("fall", fall, {})],
        [("robot.coms", 14)],
        max_steps=2,
    )
    return task


def batch(n=2):
    return {
        "context": torch.zeros(n, 14),
        "default_q": torch.zeros(n, 29),
        "default_dq": torch.zeros(n, 29),
        "hard_joint_limits": torch.tensor([-2.0, 2.0]).expand(n, 29, 2),
        "soft_joint_limits": torch.tensor([-1.8, 1.8]).expand(n, 29, 2),
        "yaw": torch.zeros(n),
        "launched": torch.ones(n, dtype=torch.bool),
        "scale": torch.ones(n, 29),
        "offset": torch.zeros(n, 29),
        "dimensions": torch.full((n, 3), 0.3),
        "task_history": torch.zeros(n, len(HISTORY_FIELDS)),
        "box_radius": torch.full((n,), 0.14),
        "launch_step": torch.zeros(n, dtype=torch.long),
        "common_step": torch.zeros(n, dtype=torch.long),
        "next_external_event_seconds": torch.full((n,), float("inf")),
    }


def test_exact_task_history_reward_gradients_and_timeout():
    task = make_task()
    b = batch()
    s = state()
    a = torch.zeros(2, 29, requires_grad=True)
    out = task.evaluate_step(
        b,
        s,
        s,
        a,
        a.detach(),
        torch.zeros(2, dtype=torch.long),
        b["task_history"],
        torch.zeros(2, dtype=torch.long),
        b["next_external_event_seconds"],
    )
    torch.testing.assert_close(
        out.rewards, torch.tensor([[0.032, 0.01, 0.0], [0.032, 0.01, 0.0]])
    )
    assert (out.history[:, -1] == 1).all()
    assert not out.terminated.any()
    assert not out.timeout.any()
    out.rewards.sum().backward()
    torch.testing.assert_close(a.grad[:, 0], torch.full((2,), 0.04))
    again = task.evaluate_step(
        b,
        s,
        s,
        a,
        a.detach(),
        out.phase,
        out.history,
        out.episode_steps,
        out.event_seconds,
        1,
    )
    assert again.timeout.all() and again.truncated.all() and not again.terminated.any()
    assert (again.history[:, -1] == 2).all()
    ending = task.evaluate_step(
        b,
        s,
        s,
        a,
        a.detach(),
        torch.full((2,), 15),
        b["task_history"],
        torch.zeros(2, dtype=torch.long),
        b["next_external_event_seconds"],
    )
    assert ending.truncated.all()
    assert (ending.phase == 15).all()


def test_pure_imagination_short_replay_and_cpu_update():
    from types import SimpleNamespace as NS
    from reflex.world_model.observation import OracleObservation
    from reflex.world_model.physics import ActionMap
    from reflex.world_model.trainer import Trainer, TrainConfig

    torch.manual_seed(3)
    torch.set_num_threads(1)
    task = make_task()
    n = 6
    b = batch(n)
    bridge = OracleObservation(
        task.fk,
        task.ref_q,
        task.ref_dq,
        task.ref_p[:, 1],
        task.ref_quat[:, 1],
        torch.zeros(29),
        torch.zeros(29),
        "torso_link",
        task.ref_arm_columns,
        task.ref_arm_down,
    )
    disc = torch.nn.Linear(166, 1)
    prior = NS(
        _disc=disc,
        _w_task=1.0,
        _w_style=1.0,
        ramp=0.1,
        reward=lambda a, b: disc(torch.cat((a, b), -1)).squeeze(-1),
    )
    cfg = TrainConfig(
        wm_kind="state",
        wm_hidden_dims=[16, 16, 16],
        hidden=16,
        batch=2,
        horizon=4,
        actor_steps=1,
        gradient_audit_every=1,
    )
    t = Trainer(
        bridge,
        ActionMap([]),
        prior,
        14,
        6,
        ["height", "persistent", "disabled"],
        {"test": "task"},
        cfg=cfg,
        task=task,
    )
    for i in range(3):
        t.replay.add(
            {
                **b,
                "state": state(n),
                "memory": torch.zeros(n, 6),
                "previous_action": torch.zeros(n, 29),
                "phase": torch.full((n,), i),
                "reference_length": torch.full((n,), 16),
                "episode_steps": torch.zeros(n, dtype=torch.long),
                "blend": torch.zeros(n, 1),
                "down": torch.empty(n, 0),
                "up": torch.empty(n, 0),
                "limits": b["hard_joint_limits"],
                "mass": torch.full((n, 1), 0.18),
                "valid": torch.ones(n, dtype=torch.bool),
                "boundary": torch.full((n,), i == 2),
                "terminated": torch.full((n,), i == 2),
                "timeout": torch.zeros(n, dtype=torch.bool),
            }
        )
    assert len(t.replay.indices(32)) == 0
    imagined = t.imagine()
    assert (imagined["valid"].sum(0) == 2).all()
    assert not imagined["done"].any()
    assert imagined["truncated"][1].all()
    result = t.update_actor()
    assert t.actor_updates == 1 and result["gradient/task"] > 0
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())


def test_initial_reference_cache_is_preserved_then_advanced():
    task = make_task()
    b = batch()
    b["reference_relative_p"] = torch.full((2, 2, 3), 7.0)
    b["reference_relative_q"] = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 2, 4)
    s = state()

    def command(step):
        return task._view(
            b,
            s,
            s,
            torch.zeros(2, 29),
            torch.zeros(2, 29),
            torch.zeros(2, dtype=torch.long),
            b["task_history"],
            torch.zeros(2, dtype=torch.long),
            step,
        ).command_manager.get_term("motion")

    torch.testing.assert_close(
        command(0).body_pos_relative_w, b["reference_relative_p"]
    )
    assert not torch.equal(command(1).body_pos_relative_w, b["reference_relative_p"])
