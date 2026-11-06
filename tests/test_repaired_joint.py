"""Repair contracts on CPU; synthetic dynamics do not establish Isaac parity."""

import copy
from types import SimpleNamespace as NS
import pytest
import torch
from reflex.world_model.task import AnalyticTask, Term
from reflex.world_model.observation import OracleObservation, PrivilegedObservation
from reflex.world_model.models import TerminationHead, StateWorldModel
from reflex.world_model.physics import ActionMap
from reflex.world_model.prior import MotionPrior
from reflex.world_model.trainer import Trainer, TrainConfig
from reflex.world_model.diagnostics import calibration, gradient_audit
from reflex.world_model.joint import returns, critic_targets
from reflex.world_model.experience import ExperienceMetrics, pair_contact
from test_world_model import builder, robot_state
from test_analytic_task import batch

TRACKED = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def height(e):
    return e.scene["robot"].data.root_pos_w[:, 2] + e.action_manager.action[:, 0]


def fall(e):
    return e.scene["robot"].data.root_pos_w[:, 2] < 0.3


def repaired(tmp_path, collect=True):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    fk = builder().fk
    states = robot_state(64)
    p, q = fk(states)
    reference = {
        "q": states[:, 7:36],
        "dq": states[:, 42:71],
        "p": torch.stack([p[n] for n in TRACKED], 1),
        "quat": torch.stack([q[n] for n in TRACKED], 1),
        "v": torch.zeros(64, 14, 3),
        "w": torch.zeros(64, 14, 3),
        "arm_columns": torch.empty(0, dtype=torch.long),
        "arm_down": torch.empty(0),
    }
    bridge = OracleObservation(
        fk,
        reference["q"],
        reference["dq"],
        p["torso_link"],
        q["torso_link"],
        torch.zeros(29),
        torch.zeros(29),
        "torso_link",
        reference["arm_columns"],
        reference["arm_down"],
    )
    task = AnalyticTask(
        fk,
        TRACKED,
        fk.joint_names if hasattr(fk, "joint_names") else ["joint"] * 29,
        TRACKED,
        "torso_link",
        reference,
        [Term("height", height, {}, 1.0), Term("disabled", height, {}, 0.0)],
        [Term("fall", fall, {})],
        [("robot.coms", 98)],
        max_steps=1000,
    )
    path = tmp_path / "repair_expert.pt"
    if not path.exists():
        torch.save(
            {
                "features": torch.randn(32, 2, 83),
                "stats": {"mean": torch.zeros(83), "std": torch.ones(83)},
            },
            path,
        )
    prior = MotionPrior(str(path), "cpu", batch=4, capacity=64, disc_steps=1)
    cfg = TrainConfig(
        privileged_critic=True,
        learned_termination=True,
        actor_zero_output=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        wm_hidden_dims=[16, 16, 16],
        model_steps=2,
        model_window=2,
        batch=4,
        capacity_steps=48,
        horizon=3,
        actor_steps=1,
        actor_batch=4,
        actor_microbatch=2,
        gradient_audit_every=0,
        diagnostic_batch=2,
        normalizer_samples=16,
    )
    t = Trainer(
        bridge,
        ActionMap([]),
        prior,
        98,
        6,
        ["height", "disabled"],
        {"synthetic": True, "tracked_names": TRACKED},
        cfg=cfg,
        task=task,
    )
    if collect:
        n = 6
        b = batch(n)
        b["context"] = torch.zeros(n, 98)
        for i in range(40):
            before = robot_state(n)
            after = before.clone()
            if i % 12 == 11:
                after[1, 2] = 0.2
            raw = torch.zeros(n, 29)
            result = task.evaluate_step(
                b,
                before,
                after,
                raw,
                raw,
                torch.full((n,), i),
                b["task_history"],
                torch.full((n,), i),
                b["next_external_event_seconds"],
            )
            tr = {
                **b,
                "state": before,
                "next_state": after,
                "memory": torch.zeros(n, 6),
                "next_memory": torch.zeros(n, 6),
                "previous_action": raw,
                "action": raw,
                "target": raw,
                "phase": torch.full((n,), i),
                "reference_length": torch.full((n,), 64),
                "episode_steps": torch.full((n,), i),
                "blend": torch.zeros(n, 1),
                "next_blend": torch.zeros(n, 1),
                "down": torch.empty(n, 0),
                "up": torch.empty(n, 0),
                "limits": b["hard_joint_limits"],
                "mass": torch.full((n, 1), 0.18),
                "stage": torch.zeros(n, dtype=torch.long),
                "valid": torch.ones(n, dtype=torch.bool),
                "boundary": result.terminated,
                "terminated": result.terminated,
                "timeout": result.timeout,
                "reward_terms": result.rewards,
                "termination_features": result.termination_features,
            }
            t.replay.add(tr)
    return t


def test_privileged_networks_and_nonzero_output_initialization(tmp_path):
    t = repaired(tmp_path)
    assert [m.out_features for m in t.actor if isinstance(m, torch.nn.Linear)] == [
        512,
        256,
        128,
        29,
    ]
    assert [m.out_features for m in t.critic if isinstance(m, torch.nn.Linear)] == [
        512,
        256,
        128,
        1,
    ]
    assert t.critic[0].in_features == 292 and t.actor[0].in_features == 179
    assert torch.count_nonzero(t.actor[-1].weight) > 0
    b = t.sample(starts=True, n=3)
    assert t.critic_observation(b).shape == (3, 292)
    assert all(isinstance(m, torch.nn.ELU) for m in list(t.actor)[1:-1:2])
    with pytest.raises(ValueError, match="14"):
        PrivilegedObservation(t.bridge, TRACKED[:2])


def rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)


def test_privileged_body_frames_match_independent_rotation_matrices(tmp_path):
    t = repaired(tmp_path)
    b = t.sample(starts=True, n=2)
    s = b["state"].clone()
    s[:, 7:36] = torch.randn(2, 29) * 0.2
    s.requires_grad_(True)
    args = (
        s,
        b["previous_action"],
        b["phase"],
        b["yaw"],
        b["launched"],
        b["dimensions"],
        b["mass"],
        b["default_q"],
        b["default_dq"],
    )
    value = t.privileged_bridge(*args)
    p, q = t.bridge.fk(s)
    a = rotation(q["torso_link"]).transpose(-1, -2)
    positions = torch.stack(
        [(a @ (p[n] - p["torso_link"])[..., None]).squeeze(-1) for n in TRACKED], 1
    ).flatten(1)
    orientations = torch.stack(
        [(a @ rotation(q[n]))[..., :2].flatten(1) for n in TRACKED], 1
    ).flatten(1)
    torch.testing.assert_close(value[:, 67:109], positions, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(value[:, 109:193], orientations, atol=1e-6, rtol=1e-5)
    value.square().mean().backward()
    assert torch.isfinite(s.grad).all()


def test_repaired_fit_actor_freezes_heads_audit_is_readonly_and_checkpoint(tmp_path):
    t = repaired(tmp_path)
    metrics = t.fit_model()
    assert metrics["termination/train_bce"] > 0
    assert "termination/brier" in t.validate()
    # Synthetic control sensitivity: stationary fake data can leave a zero WM Jacobian.
    with torch.no_grad():
        t.model.net[-1].weight.normal_(0, 1e-3)
    t.termination_opt.zero_grad(set_to_none=True)
    term_before = copy.deepcopy(t.termination.state_dict())
    model_before = copy.deepcopy(t.model.state_dict())
    result = t.update_actor()
    assert result["actor/gradient_norm"] > 0 and t.actor_updates == 1
    for k, v in term_before.items():
        torch.testing.assert_close(v, t.termination.state_dict()[k], rtol=0, atol=0)
    for k, v in model_before.items():
        torch.testing.assert_close(v, t.model.state_dict()[k], rtol=0, atol=0)
    assert all(p.grad is None for p in t.termination.parameters())
    rng = torch.get_rng_state().clone()
    actor_before = copy.deepcopy(t.actor.state_dict())
    opt_before = copy.deepcopy(t.actor_opt.state_dict())
    audit = gradient_audit(t)
    assert audit["diagnostic/direct/reward/disabled/norm"] == 0
    assert audit["diagnostic/survival_only_gradient_norm"] > 0
    assert audit["diagnostic/additive_gradient_relative_error"] < 1e-4
    torch.testing.assert_close(rng, torch.get_rng_state())
    for k, v in actor_before.items():
        torch.testing.assert_close(v, t.actor.state_dict()[k], rtol=0, atol=0)
    for k, v in opt_before["state"].items():
        for name, x in v.items():
            torch.testing.assert_close(
                x, t.actor_opt.state_dict()["state"][k][name], rtol=0, atol=0
            )
    path = tmp_path / "v3.pt"
    t.save(path)
    restored = repaired(tmp_path, collect=False)
    restored.load(path)
    assert restored.actor_updates == 1 and restored.critic_dim == 292
    assert restored.termination_updates == t.cfg.model_steps
    for k, v in t.termination.state_dict().items():
        torch.testing.assert_close(v, restored.termination.state_dict()[k])
    from test_world_model import make_trainer

    with pytest.raises(ValueError, match="mismatch"):
        make_trainer(tmp_path, "state").load(path)


def test_soft_survival_gradient_and_truncation_bootstrap():
    logit = torch.tensor(0.0, requires_grad=True)
    imagined = {
        "obs": torch.ones(4, 1, 1),
        "reward": torch.ones(3, 1),
        "done": logit.sigmoid().expand(3, 1),
        "truncated": torch.tensor([[True], [False], [False]]),
    }
    critic = torch.nn.Linear(1, 1, bias=False)
    critic.weight.data.fill_(2.0)
    value = returns(imagined, critic, 0.9)
    torch.testing.assert_close(value, torch.tensor([1.9]))
    assert torch.autograd.grad(value, logit)[0] < 0
    t = NS(critic=critic, cfg=NS(horizon=3, gamma=0.9, lam=0.95))
    torch.testing.assert_close(critic_targets(t, imagined)[0], torch.tensor([1.9]))


def test_termination_bce_calibration_and_input_gradients():
    torch.manual_seed(2)
    torch.set_num_threads(1)
    f = torch.linspace(-1, 1, 128)[:, None]
    y = (f[:, 0] > 0.2).float()
    h = TerminationHead(1, (8, 8))
    h.inputs.fit(f)
    opt = torch.optim.Adam(h.parameters(), lr=0.02)
    before = calibration(h(f), y)["termination/brier"]
    for _ in range(60):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(h(f), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    result = calibration(h(f), y)
    assert result["termination/brier"] < before * 0.25
    assert (
        sum(result[f"termination/calibration_bin{i}_count"] for i in range(10)) == 128
    )
    f = f.clone().requires_grad_(True)
    h(f).sigmoid().sum().backward()
    assert f.grad.abs().sum() > 0 and torch.isfinite(f.grad).all()


def test_contact_unknown_is_not_zero_and_completed_throw_rates():
    metric = ExperienceMetrics()
    base = {
        "reward_terms": torch.ones(1, 1),
        "terminated": torch.tensor([False]),
        "timeout": torch.tensor([False]),
    }
    metric.update(base)
    assert metric.metrics()["experience/contact_measured_fraction"] == 0
    assert "experience/box_contact_steps" not in metric.metrics()
    for launch, contact, done, success in [
        (True, False, False, False),
        (False, True, False, False),
        (False, True, True, True),
    ]:
        metric.update(
            {
                **base,
                "launch_event": torch.tensor([launch]),
                "box_contact": torch.tensor([contact]),
                "contact_known": torch.tensor([True]),
                "terminated": torch.tensor([done]),
                "strict_success": torch.tensor([success]),
            }
        )
    out = metric.metrics()
    assert (
        out["experience/box_robot_contact_per_measured_throw"] == 1
        and out["experience/strict_hold_per_completed_throw"] == 1
    )
    assert out["experience/completed_episodes"] == 1
    sensor = NS(
        cfg=NS(prim_path="/Balloon", filter_prim_paths_expr=["/Robot/.*"]),
        data=NS(force_matrix_w=torch.tensor([[[[2.0, 0.0, 0.0]]]])),
    )
    contact, known = pair_contact(sensor, 1, "cpu")
    assert contact.all() and known.all()
    sensor.cfg.filter_prim_paths_expr = ["/Ground"]
    with pytest.raises(ValueError):
        pair_contact(sensor, 1, "cpu")


def test_compile_pure_dynamics_and_fk_match_outputs_and_gradients():
    torch.manual_seed(4)
    torch.set_num_threads(1)
    model = StateWorldModel(2, hidden_layers=[8, 8, 8])
    with torch.no_grad():
        model.net[-1].weight.normal_(0, 0.001)
    start = robot_state(2)
    action = torch.randn(2, 29) * 0.1
    context = torch.randn(2, 2)

    def run(call):
        state = start.clone().requires_grad_(True)
        target = action.clone().requires_grad_(True)
        x = state
        loss = 0
        for _ in range(3):
            x = call(x, target, context)
            loss = loss + x.square().mean()
        grads = torch.autograd.grad(loss, (state, target, *model.parameters()))
        return x.detach(), grads

    eager = run(model)
    compiled = run(torch.compile(model, backend="aot_eager", fullgraph=True))
    torch.testing.assert_close(eager[0], compiled[0])
    for a, b in zip(eager[1], compiled[1]):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    fk = builder().fk

    def fk_run(call):
        s = start.clone().requires_grad_(True)
        p, q = call(s)
        value = torch.cat((p["left_wrist_yaw_link"], q["torso_link"]), -1)
        return value, torch.autograd.grad(value.square().sum(), s)[0]

    a = fk_run(fk)
    b = fk_run(torch.compile(fk, backend="aot_eager", fullgraph=True))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)


def test_cpu_joint_profile_is_exported(tmp_path):
    t = repaired(tmp_path)
    t.fit_model()
    t.profile_next_actor = True
    t.profile_dir = str(tmp_path / "profiles")
    t.update_actor()
    assert list((tmp_path / "profiles").glob("*.json"))
    text = next((tmp_path / "profiles").glob("*.txt")).read_text()
    assert "actor/BPTT" in text


def test_initialization_matches_installed_rsl_mlp_and_v3_inference(tmp_path):
    from rsl_rl.networks import MLP
    from reflex.world_model.models import mlp
    from reflex.world_model.inference import build_actor

    torch.manual_seed(31)
    original = MLP(179, 29, [512, 256, 128], activation="elu")
    torch.manual_seed(31)
    ours = mlp(179, 29, layers=[512, 256, 128])
    for a, b in zip(original.parameters(), ours.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    t = repaired(tmp_path)
    t.fit_model()
    path = tmp_path / "inference.pt"
    t.save(path)
    saved = torch.load(path, weights_only=False)
    policy = build_actor(saved, 179, "cpu")
    b = t.sample(starts=True, n=3)
    torch.testing.assert_close(policy(t.observation(b)), t.act(b))


def test_experience_resume_does_not_join_unrestored_physics():
    m = ExperienceMetrics()
    tr = {
        "reward_terms": torch.ones(1, 1),
        "terminated": torch.tensor([False]),
        "timeout": torch.tensor([False]),
    }
    m.update(tr)
    state = m.state_dict()
    restored = ExperienceMetrics()
    restored.load_state_dict(state)
    restored.update({**tr, "terminated": torch.tensor([True])})
    assert restored.metrics()["experience/episode_length"] == 1


def test_privileged_critic_without_termination_roundtrip(tmp_path):
    t = repaired(tmp_path, collect=False)
    t.cfg.learned_termination = False
    t.termination = t.termination_opt = None
    path = tmp_path / "critic_only.pt"
    t.save(path)
    other = repaired(tmp_path, collect=False)
    other.cfg.learned_termination = False
    other.termination = other.termination_opt = None
    other.load(path)
    assert other.critic_dim == 292 and other.termination is None


def test_clean2_interception_requires_five_consecutive_steps():
    from reflex.world_model.experience import clean2_intercept_candidate

    torso = torch.tensor([[0.0, 0.0, 0.8]])
    box = torch.tensor([[0.3, 0.0, 0.8]])
    v = torch.zeros(1, 3)
    assert clean2_intercept_candidate(box, v, torso, torch.tensor([0.1])).all()
    assert not clean2_intercept_candidate(
        box, torch.ones_like(v), torso, torch.tensor([0.1])
    ).any()
    m = ExperienceMetrics()
    base = {
        "reward_terms": torch.ones(1, 1),
        "terminated": torch.tensor([False]),
        "timeout": torch.tensor([False]),
    }
    m.update({**base, "launch_event": torch.tensor([True])})
    for _ in range(4):
        m.update({**base, "intercept_candidate": torch.tensor([True])})
    assert not m.throw_intercepted.any()
    m.update(
        {
            **base,
            "intercept_candidate": torch.tensor([True]),
            "terminated": torch.tensor([True]),
        }
    )
    assert m.metrics()["experience/clean2_interception_per_completed_throw"] == 1


def test_substep_contact_history_preserves_brief_collision():
    cfg = NS(
        prim_path="/Balloon", filter_prim_paths_expr=["/Robot/.*"], history_length=4
    )
    history = torch.zeros(1, 4, 1, 1, 3)
    history[:, 2, ..., 0] = 2
    sensor = NS(
        cfg=cfg,
        data=NS(force_matrix_w=torch.zeros(1, 1, 1, 3), force_matrix_w_history=history),
    )
    contact, known = pair_contact(sensor, 1, "cpu")
    assert contact.all() and known.all()


def test_cached_compiled_imagination_matches_eager_actor_gradients(tmp_path):
    t = repaired(tmp_path)
    t.fit_model()
    with torch.no_grad():
        t.model.net[-1].weight.normal_(0, 0.001)
    b = t.sample(starts=True, n=2)

    def run():
        value = t.imagine(start_batch=b)
        loss = -returns(value, t.critic, t.cfg.gamma).mean()
        grads = torch.autograd.grad(loss, tuple(t.actor.parameters()))
        return value, grads

    eager, eg = run()
    t._fk = torch.compile(t.bridge.fk, backend="aot_eager", fullgraph=True)
    t._dynamics = torch.compile(t.model, backend="aot_eager", fullgraph=True)
    compiled, cg = run()
    for key in ("obs", "critic_obs", "reward", "done"):
        torch.testing.assert_close(eager[key], compiled[key], atol=1e-6, rtol=1e-4)
    for x, y in zip(eg, cg):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)


def test_known_expensive_diagnostic_is_skipped_near_wall_deadline():
    from reflex.world_model.diagnostics import diagnostic_due

    t = NS(cfg=NS(diagnostic_every=50), actor_updates=50, last_diagnostic_seconds=600)
    assert diagnostic_due(t, 1750, 1800) == (False, True)
    assert diagnostic_due(t, 100, 1800) == (True, False)
    assert diagnostic_due(t, 1750, None) == (True, False)
    t.actor_updates = 51
    assert diagnostic_due(t, 1750, 1800) == (False, False)


def test_optional_diagnostic_history_tolerates_partial_or_invalid_records(tmp_path):
    from reflex.world_model.diagnostics import saved_diagnostic_duration

    p = tmp_path / "metrics.jsonl"
    p.write_text(
        '{"time/gradient_diagnostic_seconds":12.5}\n{"time/gradient_diagnostic_seconds":NaN}\n{"iteration":'
    )
    assert saved_diagnostic_duration({}, tmp_path / "model.pt") == 12.5
    assert (
        saved_diagnostic_duration(
            {"last_diagnostic_seconds": float("nan")}, tmp_path / "model.pt"
        )
        == 12.5
    )
    p.unlink()
    assert saved_diagnostic_duration({}, tmp_path / "model.pt") == 0.0


def test_sensor_filter_migration_keeps_physics_and_marks_old_contact_unknown(tmp_path):
    t = repaired(tmp_path)
    for key in ("contact_known", "box_contact"):
        t.replay.data[key] = torch.ones_like(t.replay.data["valid"])
    for key in ("contact_phase", "start_contact_phase"):
        t.replay.data[key] = torch.ones_like(t.replay.data["stage"])
    t.contact_filter_schema = "old_glob"
    path = tmp_path / "old_sensor.pt"
    t.save(path)
    other = repaired(tmp_path, collect=False)
    other.contact_filter_schema = "explicit_urdf_rigid_links_v2"
    other.load(path)
    assert not other.replay.data["contact_known"].any()
    assert (other.replay.data["contact_phase"] == -1).all()
    torch.testing.assert_close(t.replay.ordered("state"), other.replay.ordered("state"))
    for k, v in t.actor.state_dict().items():
        torch.testing.assert_close(v, other.actor.state_dict()[k])
    assert other.contact_filter_migration["experience_totals_reset"]


def test_disabled_contact_monitor_has_no_contact_metrics():
    metrics = ExperienceMetrics(monitor_contacts=False)
    done = torch.ones(2, dtype=torch.bool)
    metrics.update(
        {
            "reward_terms": torch.ones(2, 1),
            "terminated": done,
            "timeout": ~done,
            "strict_success": done,
            "contact_known": done,
            "box_contact": done,
        }
    )
    result = metrics.metrics()
    assert not any("contact" in key for key in result)
    assert result["experience/completed_strict_successes"] == 2


def test_diagnostic_restores_compiled_callables_on_failure():
    from reflex.world_model.diagnostics import eager_diagnostic

    compiled_model, compiled_fk = object(), object()
    eager_model, eager_fk = object(), object()
    t = NS(
        _dynamics=compiled_model,
        _fk=compiled_fk,
        model=eager_model,
        bridge=NS(fk=eager_fk),
    )
    with pytest.raises(RuntimeError):
        with eager_diagnostic(t):
            assert t._dynamics is eager_model and t._fk is eager_fk
            raise RuntimeError("audit failure")
    assert t._dynamics is compiled_model and t._fk is compiled_fk


def test_expanded_compile_matches_actor_gradients(tmp_path):
    t = repaired(tmp_path)
    t.fit_model()
    b = t.sample(starts=True, n=2)

    def run():
        v = t.imagine(start_batch=b)
        g = torch.autograd.grad(
            -returns(v, t.critic, t.cfg.gamma).mean(), tuple(t.actor.parameters())
        )
        return v, g

    eager, eg = run()
    t._transition = torch.compile(t.transition, backend="aot_eager", fullgraph=True)
    t._observations = torch.compile(
        t.imagined_observations, backend="aot_eager", fullgraph=True
    )
    actual, ag = run()
    for key in ("obs", "critic_obs", "reward", "done", "truncated"):
        torch.testing.assert_close(eager[key], actual[key], atol=1e-6, rtol=1e-4)
    for x, y in zip(eg, ag):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-4)


def test_predicted_termination_features_match_imagination(tmp_path):
    t = repaired(tmp_path)
    b = t.sample(starts=True, n=3)
    s = b["state"]
    ns = t.model(s, b["target"], b["context"])
    for step in (0, 3):
        actual = t.task.evaluate_step(
            b,
            s,
            ns,
            b["action"],
            b["previous_action"],
            b["phase"],
            b["task_history"],
            b["episode_steps"],
            b["next_external_event_seconds"],
            step,
        )
        features = t.task.predicted_termination_features(
            b, s, ns, b["phase"], initial=step == 0
        )
        torch.testing.assert_close(features, actual.termination_features)


def test_joint_predicted_bce_updates_wm_and_head(tmp_path):
    t = repaired(tmp_path)
    t.cfg.joint_predicted_termination = True
    t.cfg.state_loss_reduction = "sum"
    t._termination_probe_due = True
    t.cfg.variable_model_windows = True
    metrics = t.fit_model()
    assert metrics["termination/probe_weighted_bce_wm_gradient_norm"] > 0
    assert t.model_updates == t.termination_updates == t.cfg.model_steps
    assert metrics["wm/loss"] == pytest.approx(
        metrics["wm/state_loss"] + metrics["termination/train_bce"], rel=1e-5
    )


def test_head_training_migration_is_explicit_and_preserves_adam(tmp_path):
    t = repaired(tmp_path)
    t.fit_model()
    path = tmp_path / "separate.pt"
    t.save(path)
    new = repaired(tmp_path, collect=False)
    new.cfg.joint_predicted_termination = True
    new.cfg.state_loss_reduction = "sum"
    with pytest.raises(ValueError, match="mismatch"):
        new.load(path)
    new.load(path, migrate_termination_training=True)
    assert new.initialization["training_migration"]["optimizers_preserved"]
    for old_opt, new_opt in (
        (t.model_opt, new.model_opt),
        (t.termination_opt, new.termination_opt),
    ):
        for key, value in old_opt.state_dict()["state"].items():
            for name, expected in value.items():
                torch.testing.assert_close(
                    expected, new_opt.state_dict()["state"][key][name]
                )
    new.fit_model()
    assert new.termination_updates == new.model_updates == 2 * new.cfg.model_steps
