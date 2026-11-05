"""CPU contracts: frames/FK, replay boundaries, pathwise chain and full resume.

Synthetic dynamics are deliberately identified as synthetic. These tests cannot
establish real Isaac / PhysX parity or catching performance.
"""

import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
import torch
from reflex.world_model.physics import (
    Kinematics,
    ActionMap,
    amp_features,
    canonical,
    expmap,
    integrate,
    inverse,
    mul,
    apply,
    rot6,
    state_error,
)
from reflex.world_model.observation import OracleObservation
from reflex.world_model.replay import Replay
from reflex.world_model.prior import MotionPrior
from reflex.world_model.trainer import Trainer, TrainConfig, lambda_returns

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "assets/robots/main_hands.urdf"


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    np.random.seed(17)


def robot_state(n=6):
    s = torch.zeros(n, 84)
    s[:, 2] = 0.8
    s[:, 3] = s[:, 74] = 1
    s[:, 71] = 0.6
    s[:, 73] = 1.2
    return s


def builder():
    names = [
        j.get("name")
        for j in ET.parse(URDF).getroot().findall("joint")
        if j.get("type") != "fixed"
    ]
    assert len(names) == 29
    fk = Kinematics(URDF, names)
    z = torch.zeros(512, 29)
    p = torch.zeros(512, 3)
    p[:, 2] = 0.8
    q = torch.zeros(512, 4)
    q[:, 0] = 1
    return OracleObservation(
        fk,
        z,
        z,
        p,
        q,
        torch.zeros(29),
        torch.zeros(29),
        "torso_link",
        torch.tensor(
            [
                i
                for i, n in enumerate(names)
                if any(k in n for k in ("shoulder", "elbow", "wrist"))
            ]
        ),
        torch.zeros(14),
    )


def test_rotations_fk_and_heading_invariance():
    s = robot_state(2)
    s[:, 7:36] = torch.randn(2, 29) * 0.1
    bridge = builder()
    s.requires_grad_()
    pos, quat = bridge.fk(s)
    assert "left_wrist_yaw_link" in pos
    pos["left_wrist_yaw_link"].sum().backward()
    assert s.grad[:, 7:36].abs().sum() > 0
    h = expmap(torch.tensor([[0.0, 0.0, 0.7]])).expand(2, 4)
    moved = s.detach().clone()
    for cols in (slice(0, 3), slice(71, 74)):
        moved[:, cols] = apply(h, s.detach()[:, cols]) + torch.tensor([5.0, -2.0, 0.0])
    for cols in (slice(36, 39), slice(39, 42), slice(78, 81), slice(81, 84)):
        moved[:, cols] = apply(h, s.detach()[:, cols])
    moved[:, 3:7] = mul(h, s.detach()[:, 3:7])
    moved[:, 74:78] = mul(h, s.detach()[:, 74:78])
    torch.testing.assert_close(
        canonical(moved), canonical(s.detach()), atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        amp_features(moved, bridge.fk),
        amp_features(s.detach(), bridge.fk),
        atol=2e-6,
        rtol=1e-5,
    )
    a = expmap(torch.tensor([[0.3, -0.2, 0.6]]))
    v = torch.randn(1, 3)
    torch.testing.assert_close(apply(inverse(a), apply(a, v)), v)
    assert rot6(a).shape == (1, 6)


def test_integrator_quaternion_and_gradient():
    s = robot_state(2)
    s[:, 36] = 0.5
    s[:, 81] = 1
    d = torch.randn(2, 82, requires_grad=True) * 0.1
    ns = integrate(s, d, 0.02)
    torch.testing.assert_close(ns[:, 3:7].norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(ns[:, 74:78].norm(dim=-1), torch.ones(2))
    opposite = ns.clone()
    opposite[:, 3:7] *= -1
    opposite[:, 74:78] *= -1
    torch.testing.assert_close(
        state_error(ns, opposite), torch.zeros(2, 82), atol=1e-5, rtol=0
    )
    assert torch.isfinite(torch.autograd.grad(ns.square().sum(), d)[0]).all()


def test_action_map_state_limits_and_wrist_lock():
    amap = ActionMap([0, 1], ramp=8, delta_min=0.15)
    raw = torch.ones(2, 29, requires_grad=True)
    scale = torch.ones(2, 29)
    scale[:, 2] = 0
    off = torch.zeros(2, 29)
    off[:, 2] = 1.57
    limits = torch.tensor([-2.0, 2.0]).expand(2, 29, 2)
    target, b = amap(
        raw,
        torch.zeros(2, 1),
        torch.tensor([True, False]),
        scale,
        off,
        torch.full((2, 2), -0.2),
        torch.full((2, 2), 0.4),
        limits,
    )
    torch.testing.assert_close(b, torch.tensor([[0.125], [0.0]]))
    torch.testing.assert_close(target[:, 2], torch.full((2,), 1.57))
    target.sum().backward()
    assert raw.grad[:, 2].abs().sum() == 0
    assert raw.grad[:, 0].min() > 0


def test_replay_no_crossing_events_timeouts_or_wrap():
    r = Replay(8)
    for t in range(12):
        r.add(
            {
                "valid": torch.ones(6, dtype=torch.bool),
                "boundary": torch.full((6,), t == 8),
                "terminated": torch.zeros(6, dtype=torch.bool),
                "timeout": torch.full((6,), t == 10),
                "launched": torch.full((6,), t >= 6),
                "phase": torch.full((6,), t),
                "reference_length": torch.full((6,), 100),
                "label": torch.full((6,), t),
            }
        )
    seq = r.sample(64, 3)
    assert torch.all(seq["label"][1:] - seq["label"][:-1] == 1)
    assert not seq["boundary"][:-1].any() and not seq["timeout"][:-1].any()
    assert (
        (seq["label"][:, 0] == torch.tensor([4, 5, 6])).all()
        or seq["label"][0, 0] == 5
        or seq["label"][0, 0] == 6
    )
    starts = r.sample(20, 2, starts=True)
    assert starts["launched"][0].all()
    assert not (starts["phase"][0] == 8).any()
    v = r.sample(10, 1, validation=True)
    assert v["phase"].shape == (1, 10)


def test_timeout_lambda_bootstrap():
    r = torch.tensor([[1.0], [2.0]])
    v = torch.tensor([[0.0], [3.0], [10.0]])
    actual = lambda_returns(r, torch.zeros_like(r), v, 1.0, 1.0)
    torch.testing.assert_close(actual, torch.tensor([[13.0], [12.0]]))
    actual = lambda_returns(r, torch.tensor([[0.0], [1.0]]), v, 1.0, 1.0)
    torch.testing.assert_close(actual, torch.tensor([[3.0], [2.0]]))


def test_com_velocity_integrates_link_origin():
    s = robot_state(1)
    s[:, 41] = 2.0
    com = torch.tensor([[0.1, 0.0, 0.0]])
    # COM travels along +y when the link origin rotates in place.
    s[:, 37] = 0.2
    ns = integrate(s, torch.zeros(1, 82), 0.02, root_com=com)
    torch.testing.assert_close(ns[:, :3], s[:, :3])


def test_observation_per_environment_default_and_inference(tmp_path):
    from reflex.world_model.inference import build_actor

    t = make_trainer(tmp_path)
    b = SyntheticAdapter(t.bridge).snapshot()
    original = t.observation(b)
    b["default_q"] = torch.ones(6, 29) * 0.01
    altered = t.observation(b)
    # q-relative channels follow command58 + anchor9 + root velocities6.
    torch.testing.assert_close(altered[:, 73:102], original[:, 73:102] - 0.01)
    path = tmp_path / "policy.pt"
    t.save(path)
    checkpoint = torch.load(path, weights_only=False)
    policy = build_actor(checkpoint, 179, "cpu")
    with torch.no_grad():
        torch.testing.assert_close(policy(altered), t.act(b))
    with pytest.raises(ValueError, match="179-D"):
        build_actor(checkpoint, 170, "cpu")


def test_reward_features_distinguish_raw_history_and_reference(tmp_path):
    t = make_trainer(tmp_path)
    b = SyntheticAdapter(t.bridge).snapshot()
    raw = torch.zeros(6, 29)
    changed = raw.clone()
    changed[:, 2] = 1.0
    b["scale"][:, 2] = 0.0
    target, blend = t.targets(b, raw)
    same_target, _ = t.targets(b, changed)
    torch.testing.assert_close(target, same_target)

    def features(batch, action):
        return t.model.features(
            batch["state"],
            target,
            batch["context"],
            batch["memory"],
            batch["phase"],
            blend,
            t.observation(batch),
            action,
        )

    first = features(b, raw)
    assert not torch.equal(first, features(b, changed))
    previous = {**b, "previous_action": changed}
    assert not torch.equal(first, features(previous, raw))
    moved = {**b, "state": b["state"].clone()}
    moved["state"][:, :2] += 1.0
    moved["state"][:, 71:73] += 1.0
    # Canonical physical state loses this displacement, but the fixed reference
    # does not move: reward conditioning must preserve the anchor discrepancy.
    torch.testing.assert_close(canonical(b["state"]), canonical(moved["state"]))
    assert not torch.equal(first, features(moved, raw))


def test_warm_actor_projection_and_normalizer_preserve_policy(tmp_path):
    from reflex.world_model.models import mlp

    t = make_trainer(tmp_path)
    old = mlp(170, 29, 32)
    source = tmp_path / "ppo.pt"
    torch.save(
        {"model_state_dict": {"actor." + k: v for k, v in old.state_dict().items()}},
        source,
    )
    t.warm_start_actor(source)
    adapter = SyntheticAdapter(t.bridge)
    t.collect(adapter)
    b = {k: v[0] for k, v in t.sample().items()}
    obs = t.observation(b)
    with torch.no_grad():
        expected = old(torch.cat((obs[:, :166], torch.zeros(len(obs), 4)), -1))
        torch.testing.assert_close(t.act(b), expected)
        t.fit_normalizers()
        torch.testing.assert_close(t.act(b), expected, atol=2e-6, rtol=1e-5)
    assert t.initialization["kind"] == "ppo_actor_warm_start"
    assert not t.initialization["optimizer_replay_amp_restored"]


def test_balanced_sampler_retains_holdout_and_event_boundaries():
    r = Replay(40)
    for step in range(40):
        stage = 0 if step == 10 else 3
        r.add(
            {
                "valid": torch.ones(6, dtype=torch.bool),
                "boundary": torch.full((6,), step == 20),
                "terminated": torch.zeros(6, dtype=torch.bool),
                "timeout": torch.zeros(6, dtype=torch.bool),
                "launched": torch.ones(6, dtype=torch.bool),
                "phase": torch.full((6,), step),
                "reference_length": torch.full((6,), 100),
                "stage": torch.full((6,), stage),
                "environment": torch.arange(6),
            }
        )
    batch = r.sample(1000, 4, stage_balanced=True)
    assert int((batch["stage"][0] == 0).sum()) > 400
    assert not (batch["environment"] % 5 == 0).any()
    assert not batch["boundary"][:-1].any()
    held = r.sample(1000, 4, validation=True, stage_balanced=True)
    assert (held["environment"] % 5 == 0).all()
    assert int((held["stage"][0] == 0).sum()) < 100


def test_wm_offline_uses_recorded_actions_without_actor(tmp_path):
    from reflex.world_model.offline import rollout, window_loss

    t = make_trainer(tmp_path)
    t.collect(SyntheticAdapter(t.bridge))
    t.fit_normalizers()

    def forbidden(*args, **kwargs):
        raise AssertionError("Actor called during WM-only prediction")

    t.actor.forward = forbidden
    seq = t.replay.sample(4, 8)
    states, *_ = rollout(t.model, t.bridge, seq, 8)
    assert states.shape == (8, 4, 84)
    loss = window_loss(t.model, t.bridge, seq, 8)
    t.model_opt.zero_grad(set_to_none=True)
    loss.backward()
    assert any(p.grad is not None for p in t.model.parameters())
    assert all(p.grad is None for p in t.actor.parameters())


class SyntheticAdapter:
    """Analytic toy transitions, never claimed to emulate real box contact."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.s = robot_state()
        self.phase = 0
        self.previous = torch.zeros(6, 29)
        self.blend = torch.zeros(6, 1)

    def snapshot(self):
        self.before = {
            "state": self.s.clone(),
            "previous_action": self.previous.clone(),
            "phase": torch.full((6,), self.phase, dtype=torch.long),
            "yaw": torch.zeros(6),
            "launched": torch.ones(6, dtype=torch.bool),
            "dimensions": torch.full((6, 3), 0.3),
            "mass": torch.full((6, 1), 0.18),
            "blend": self.blend.clone(),
            "scale": torch.ones(6, 29),
            "offset": torch.zeros(6, 29),
            "down": torch.zeros(6, 2),
            "up": torch.zeros(6, 2),
            "limits": torch.tensor([-3.0, 3.0]).expand(6, 29, 2),
            "context": torch.ones(6, 2),
            "memory": torch.zeros(6, 6),
            "stage": torch.arange(6) % 4,
            "reference_length": torch.full((6,), 512),
        }
        return self.before

    def step(self, raw, target, blend):
        b = self.before
        delta = torch.zeros(6, 82)
        delta[:, 6:35] = target * 0.4
        self.s = integrate(self.s, delta, 0.02).detach()
        rewards = torch.stack(
            (-(self.s[:, 7] - 0.5).square(), -self.s[:, 42].square() * 0.01), -1
        )
        amp = amp_features(b["state"], self.bridge.fk)
        na = amp_features(self.s, self.bridge.fk)
        self.phase += 1
        self.previous = raw.detach()
        self.blend = blend.detach()
        no = torch.zeros(6, dtype=torch.bool)
        return {
            **b,
            "action": raw,
            "target": target,
            "next_blend": blend,
            "next_state": self.s,
            "next_memory": torch.zeros(6, 6),
            "reward_terms": rewards,
            "terminated": no,
            "timeout": no,
            "boundary": no,
            "valid": ~no,
            "amp": amp,
            "next_amp": na,
        }


def make_trainer(tmp_path):
    bridge = builder()
    expert = tmp_path / "expert.pt"
    if not expert.exists():
        torch.save(
            {
                "features": torch.randn(32, 2, 83),
                "stats": {"mean": torch.zeros(83), "std": torch.ones(83)},
            },
            expert,
        )
    prior = MotionPrior(str(expert), "cpu", batch=8, capacity=256, disc_steps=1)
    c = TrainConfig(
        capacity_steps=48,
        collect_steps=40,
        model_steps=12,
        model_window=2,
        batch=8,
        horizon=4,
        actor_steps=1,
        hidden=32,
        warmup_iterations=0,
    )
    return Trainer(
        bridge,
        ActionMap([0, 1]),
        prior,
        2,
        6,
        ["target", "speed"],
        {"synthetic": True},
        cfg=c,
    )


def test_full_cpu_chain_gradients_checkpoint_resume(tmp_path):
    t = make_trainer(tmp_path)
    adapter = SyntheticAdapter(t.bridge)
    t.collect(adapter)
    assert t.real_transitions == 240
    before = copy.deepcopy(t.model.state_dict())
    metrics = t.fit_model()
    assert np.isfinite(metrics["wm/loss"])
    assert any(not torch.equal(before[k], t.model.state_dict()[k]) for k in before)
    validation = t.validate()
    assert "validation/16/state_rmse" in validation
    actor_before = copy.deepcopy(t.actor.state_dict())
    model_before = copy.deepcopy(t.model.state_dict())
    disc_before = copy.deepcopy(t.prior._disc.state_dict())
    metrics = t.update_actor()
    assert metrics["gradient/task"] > 0 and metrics["gradient/style"] > 0
    assert any(
        not torch.equal(actor_before[k], t.actor.state_dict()[k]) for k in actor_before
    )
    for k, v in model_before.items():
        torch.testing.assert_close(v, t.model.state_dict()[k], rtol=0, atol=0)
    for k, v in disc_before.items():
        torch.testing.assert_close(v, t.prior._disc.state_dict()[k], rtol=0, atol=0)
    t.iteration = 7
    # Simulate an explicit end-of-session edge. Load inserts this edge too.
    t.replay.data["boundary"][(t.replay.pointer - 1) % t.replay.capacity].fill_(True)
    path = tmp_path / "checkpoint.pt"
    t.save(path)
    # Existing v1 checkpoints predate optional sampler / actor shape fields.
    legacy = torch.load(path, weights_only=False)
    for key in ("actor_hidden_dims", "stage_balanced"):
        legacy["config"].pop(key)
    legacy.pop("initialization")
    torch.save(legacy, path)
    expected_model = t.fit_model()
    expected_actor = t.update_actor()
    expected = copy.deepcopy(t.actor.state_dict())
    restored = make_trainer(tmp_path)
    restored.load(path)
    assert restored.iteration == 7 and restored.real_transitions == 240
    numeric = lambda metrics: {
        k: v for k, v in metrics.items() if not k.startswith("time/")
    }
    assert numeric(restored.fit_model()) == numeric(expected_model)
    assert numeric(restored.update_actor()) == numeric(expected_actor)
    for k, v in expected.items():
        torch.testing.assert_close(v, restored.actor.state_dict()[k], rtol=0, atol=0)
    assert not restored.ready(validation, ranking_approved=False)
    restored.contract = {"wrong": True}
    with pytest.raises(ValueError, match="contract mismatch"):
        restored.load(path)
