"""Formula parity and controlled new-mode tests; no Isaac/GPU result claimed."""

import copy
from dataclasses import replace

import pytest
import torch
from reflex.world_model.analytic_termination import probability, soft_termination
from reflex.world_model.observation import PrivilegedObservation
from reflex.world_model.trainer import Trainer
from test_repaired_joint import repaired


def new_trainer(old, **overrides):
    cfg = replace(
        old.cfg,
        learned_termination=False,
        joint_predicted_termination=False,
        analytic_soft_termination=True,
        pure_soft_survival=True,
        **overrides,
    )
    return Trainer(
        copy.deepcopy(old.bridge),
        copy.deepcopy(old.action_map),
        old.prior,
        old.contract.get("context_dim", 98),
        6,
        old.reward_names,
        copy.deepcopy(old.contract),
        cfg=cfg,
        task=copy.deepcopy(old.task),
        event_scheduler=copy.deepcopy(old.event_scheduler),
    )


def matrix(q):
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
    ).reshape(*q.shape[:-1], 3, 3)


def reference(z, q, rz, rq):
    # Literal upstream matrix-entry formula, no Euler conversion.
    ez = (rz - z).abs()
    et = (matrix(rq)[..., 2, 2] - matrix(q)[..., 2, 2]).abs()
    return 1 - (1 - torch.sigmoid((ez - 0.25) / 0.02)) * (
        1 - torch.sigmoid((et - 0.8) / 0.04)
    )


def test_upstream_formula_and_input_gradients():
    torch.manual_seed(3)
    q = torch.nn.functional.normalize(
        torch.randn(100, 4, dtype=torch.float64), dim=-1
    ).requires_grad_()
    rq = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    z = torch.randn(100, dtype=torch.float64, requires_grad=True)
    rz = z.detach() + torch.linspace(-0.4, 0.4, 100, dtype=torch.float64)
    a, b = soft_termination(z, q, rz, rq), reference(z, q, rz, rq)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(a.sum(), (z, q), retain_graph=True)
    gb = torch.autograd.grad(b.sum(), (z, q))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert torch.isfinite(x).all()


def test_threshold_probability_and_continuous_gradient():
    z = torch.tensor(
        [0.25 - 1e-6, 0.25, 0.25 + 1e-6], dtype=torch.float64, requires_grad=True
    )
    q = torch.tensor([[1.0, 0, 0, 0]], dtype=z.dtype).expand(3, -1)
    p = soft_termination(z, q, torch.zeros_like(z), q)
    grad = torch.autograd.grad(p.sum(), z)[0]
    assert torch.isfinite(grad).all() and (grad > 0).all()
    torch.testing.assert_close(
        p[1], torch.tensor(0.5, dtype=z.dtype), atol=2e-9, rtol=0
    )
    assert (grad - grad[1]).abs().max() < 1e-6


def test_critic_includes_every_actor_box_column(tmp_path):
    old = repaired(tmp_path)
    new = new_trainer(old, critic_include_all_actor=True)
    b = old.sample(starts=True, n=4)
    actor = new.observation(b)
    original = PrivilegedObservation(new.bridge, old.contract["tracked_names"])(
        b["state"],
        b["previous_action"],
        b["phase"],
        b["yaw"],
        b["launched"],
        b["dimensions"],
        b["mass"],
    )
    value = new.privileged_bridge(
        b["state"],
        b["previous_action"],
        b["phase"],
        b["yaw"],
        b["launched"],
        b["dimensions"],
        b["mass"],
    )
    assert value.shape == (4, 305) and actor.shape == (4, 179)
    torch.testing.assert_close(value[:, :292], original, rtol=0, atol=0)
    torch.testing.assert_close(value[:, 292:], actor[:, 166:], rtol=0, atol=0)
    assert new.critic[0].in_features == 305


def test_h8_no_head_bce_and_explicit_checkpoint_migration(tmp_path):
    old = repaired(tmp_path)
    old.cfg.pure_soft_survival = True
    old.cfg.model_window = 8
    old.cfg.horizon = 32
    old.fit_model()
    path = tmp_path / "learned.pt"
    old.save(path)
    new = new_trainer(
        old,
        critic_include_all_actor=True,
        batch_imagination_amp=True,
        batch_imagination_critic=True,
    )
    with pytest.raises(ValueError, match="mismatch"):
        new.load(path)
    new.load(
        path,
        migrate_analytic_soft_termination=True,
        migrate_critic_box_observation=True,
    )
    assert new.termination is new.termination_opt is None
    assert not new.critic_opt.state
    b = old.sample(starts=True, n=4)
    torch.testing.assert_close(
        old.critic(old.critic_observation(b)), new.critic(new.critic_observation(b))
    )
    for name in ("actor", "model"):
        for k, v in getattr(old, name).state_dict().items():
            torch.testing.assert_close(
                v, getattr(new, name).state_dict()[k], rtol=0, atol=0
            )
    metrics = new.fit_model()
    assert "termination/train_bce" not in metrics and new.termination_updates == 0
    assert metrics["wm/loss"] == pytest.approx(metrics["wm/state_loss"])
    result = new.imagine(b)
    assert result["done"].shape == (32, 4)
    gradient = torch.autograd.grad(result["done"].sum(), tuple(new.actor.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient)
    update = new.update_actor()
    assert update["actor/gradient_norm"] > 0
    target = tmp_path / "analytic.pt"
    new.save(target)
    clone = new_trainer(old, critic_include_all_actor=True)
    clone.load(target)
    assert clone.actor_updates == new.actor_updates
    assert clone.termination is None and clone.termination_updates == 0
    with pytest.raises(ValueError, match="mismatch"):
        old.load(target)


def test_saturated_probability_does_not_hard_stop_h32(tmp_path):
    old = repaired(tmp_path)
    new = new_trainer(old)
    new.cfg.horizon = 32
    b = old.sample(starts=True, n=4)
    b["state"][:, 2] = 5.0
    b["phase"].zero_()
    b["episode_steps"].zero_()
    b["reference_length"].fill_(64)
    b["next_external_event_seconds"].fill_(100.0)
    new.termination = None
    result = new.imagine(b)
    assert result["done"][0].eq(1).all()
    assert result["valid"].all()  # High probability is not a discrete boundary.


def test_next_reference_frame_is_used(tmp_path):
    old = repaired(tmp_path)
    new = new_trainer(old)
    root = new.task.tracked_names.index("pelvis")
    new.task.ref_p[0, root, 2] = 0.8
    new.task.ref_p[1, root, 2] = 1.05
    physical = torch.zeros(1, 84)
    physical[:, 2] = 0.8
    physical[:, 3] = 1.0
    first = probability(new.task, physical, torch.tensor([0]))
    next_value = probability(new.task, physical, torch.tensor([1]))
    assert first.item() < 1e-5
    torch.testing.assert_close(next_value, torch.tensor([0.5]))


def test_native_launch_h8_fused_physics_only_and_compiled_formula(tmp_path):
    from test_compact_context import compact_trainer, fill

    old, b = compact_trainer(tmp_path, mode="full")
    fill(old, b)
    old.cfg.pure_soft_survival = True
    old.cfg.horizon = 32
    path = tmp_path / "native_learned.pt"
    old.save(path)
    new = new_trainer(old, critic_include_all_actor=True)
    new.load(
        path,
        migrate_analytic_soft_termination=True,
        migrate_critic_box_observation=True,
    )
    new.cfg.compile_wm_training_step = True
    new._wm_training_step = torch.compile(
        new.v24_model_training_step, fullgraph=True, backend="aot_eager"
    )
    metrics = new.fit_model()
    assert "termination/train_bce" not in metrics
    assert new.termination_updates == 0 and new.termination is None
    assert metrics["wm/loss"] == pytest.approx(metrics["wm/physical_state_loss"])
    start = {
        k: v[0].clone()
        for k, v in new.replay.data.items()
        if k not in new.replay.global_fields
    }
    start["event_pool"] = (
        new.replay.data["event_pool"][0].unsqueeze(0).expand(4, -1, -1).clone()
    )
    start["state"][:, 2] = 5.0
    result = new.imagine(start, trace=True)
    assert result["done"][0].eq(1).all()
    assert result["valid"].all()
    assert not result["predicted_launch_probability"][0].bool().any()
    assert result["predicted_launch_probability"][1].bool().all()
    physical = result["physical_states"]
    phase = result["reference_next_phases"]
    torch.testing.assert_close(
        result["done"], probability(new.task, physical, phase), atol=0, rtol=0
    )
    z = torch.tensor([0.25], requires_grad=True)
    q = torch.tensor([[1.0, 0, 0, 0]])
    compiled = torch.compile(soft_termination, fullgraph=True, backend="aot_eager")
    actual = compiled(z, q, torch.zeros_like(z), q)
    expected = soft_termination(z, q, torch.zeros_like(z), q)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), z)[0],
        torch.autograd.grad(expected.sum(), z)[0],
        atol=0,
        rtol=0,
    )


def test_paired_soft_probe_restores_source_and_reports_formula(tmp_path):
    from reflex.world_model.soft_termination_check import check

    old = repaired(tmp_path)
    old.cfg.pure_soft_survival = True
    old.cfg.horizon = 32
    before = {k: v.clone() for k, v in old.actor.state_dict().items()}
    rng = torch.get_rng_state().clone()
    original_cfg = old.cfg
    report = check(old, tmp_path, repeats=1)
    assert old.cfg is original_cfg
    assert report["completed"]
    assert report["real_states"]["formula_max_abs"] == 0
    assert report["variants"]["analytic"]["trajectory_formula_max_abs"] == 0
    assert report["variants"]["analytic"]["gradient_finite"]
    assert old.termination is not None and old.cfg.learned_termination
    assert not old.cfg.analytic_soft_termination and old.actor_updates == 0
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    for k, v in old.actor.state_dict().items():
        torch.testing.assert_close(v, before[k], atol=0, rtol=0)


def test_tilt_Rzz_threshold_and_gradient():
    angle = torch.acos(torch.tensor(0.2, dtype=torch.float64)).requires_grad_()
    q = torch.stack(
        (torch.cos(angle / 2), torch.sin(angle / 2), angle * 0, angle * 0)
    ).unsqueeze(0)
    rq = torch.tensor([[1.0, 0, 0, 0]], dtype=torch.float64)
    z = torch.zeros(1, dtype=torch.float64)
    p = soft_termination(z, q, z, rq)
    torch.testing.assert_close(p, torch.tensor([0.5], dtype=p.dtype), atol=2e-6, rtol=0)
    gradient = torch.autograd.grad(p.sum(), angle)[0]
    assert torch.isfinite(gradient) and gradient > 0
