import copy
import torch
from reflex.world_model.trainer import Trainer
from reflex.world_model.joint import returns
from test_repaired_joint import repaired


def macro_from_old(tmp_path):
    old = repaired(tmp_path)
    old.fit_model()
    path = tmp_path / "old.pt"
    old.save(path)
    cfg = copy.deepcopy(old.cfg)
    cfg.learn_external_events = True
    cfg.joint_predicted_termination = True
    cfg.state_loss_reduction = "sum"
    cfg.variable_model_windows = True
    cfg.diagnostic_every = 0
    new = Trainer(
        copy.deepcopy(old.bridge),
        copy.deepcopy(old.action_map),
        old.prior,
        98,
        6,
        old.reward_names,
        old.contract,
        cfg=cfg,
        task=copy.deepcopy(old.task),
    )
    new.warm_start_macro(path)
    for i in range(old.replay.count):
        tr = {k: old.replay.ordered(k)[i].clone() for k in old.replay.data}
        tr["physical_next_state"] = tr["next_state"].clone()
        tr["next_state"] = tr["next_state"].clone()
        if i == 9:
            tr["next_state"][:, 78] = 4.0
        tr["launched"] = torch.full((6,), i >= 10)
        tr["after_launched"] = torch.full((6,), i >= 9)
        tr["external_boundary"] = torch.full((6,), i == 9)
        new.replay.add(tr)
    return old, new


def test_macro_warm_start_and_external_write_supervision(tmp_path):
    old, new = macro_from_old(tmp_path)
    b = old.sample(starts=True, n=3)
    physical, post, _ = new.model.predict_all(
        b["state"], b["target"], b["context"], b["launched"]
    )
    torch.testing.assert_close(
        physical, old.model(b["state"], b["target"], b["context"])
    )
    torch.testing.assert_close(post, physical)
    assert new.model_opt.state_dict()["state"] == {}
    assert new.initialization["replay_reset"]
    # The written 4 m/s target is kept, separately from pre-write physics.
    assert new.replay.ordered("next_state")[9, :, 78].eq(4).all()
    assert not new.replay.ordered("physical_next_state")[9, :, 78].eq(4).any()
    ix = new.replay.indices(8)
    assert ((ix[:, 0] == 5) & (ix[:, 1] == 2)).any()
    metrics = new.fit_model()
    expected = (
        metrics["wm/physical_state_loss"]
        + metrics["wm/post_control_state_loss"]
        + metrics["wm/launch_mode_bce"]
        + metrics["termination/train_bce"]
    )
    assert abs(metrics["wm/loss"] - expected) < 1e-4
    assert metrics["termination/probe_weighted_bce_wm_gradient_norm"] > 0
    assert new.model_updates == new.termination_updates == new.cfg.model_steps
    assert torch.isfinite(torch.tensor(list(new.validate().values()))).all()
    path = tmp_path / "macro.pt"
    new.save(path)
    new.load(path)


def test_macro_waiting_imagination_uses_learned_mode_not_event_scheduler(tmp_path):
    _, new = macro_from_old(tmp_path)
    b = new.sample(starts=True, n=3)
    b["launched"].zero_()
    b["next_external_event_seconds"].fill_(0.001)
    with torch.no_grad():
        new.model.net[-1].bias[-1] = 10
    imagined = new.imagine(start_batch=b)
    assert not imagined["truncated"].any()
    assert imagined["predicted_launch_probability"].min() > 0.99
    g = torch.autograd.grad(
        -returns(imagined, new.critic, new.cfg.gamma).mean(),
        tuple(new.actor.parameters()),
    )
    assert sum(float(x.square().sum()) for x in g) > 0
