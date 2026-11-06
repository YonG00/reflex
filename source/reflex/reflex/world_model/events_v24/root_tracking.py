"""Physical/root-height and terminal-feature diagnostics in recorded-event rollouts."""

import json
from pathlib import Path

import torch


@torch.no_grad()
def check(t, output):
    rows = {}
    for h in (1, 8, 16, 32):
        try:
            seq = t.replay.sample(128, h, validation=True)
        except ValueError:
            rows[str(h)] = {"available": False}
            continue
        s = seq["state"][0]
        errors, anchors, joints, velocities, probabilities, labels = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        anchor_sign_disagreement = []
        for j in range(h):
            b = {k: v[j] for k, v in seq.items()}
            pre = t.event_scheduler.codec.pre_features(
                seq["event_state"][j], seq["episode_steps"][j], t.task.dt
            )
            physical = t.model.predict_physical(
                s, seq["target"][j], seq["context"][j], pre
            )
            true = seq["physical_next_state"][j]
            feat = t.task.predicted_termination_features(
                b, s, physical, seq["phase"][j]
            )
            true_feat = seq["termination_features"][j]
            errors.append((physical[:, 2] - true[:, 2]).square())
            anchors.append((feat[:, 2] - true_feat[:, 2]).square())
            anchor_sign_disagreement.append(
                (feat[:, 2].abs() > 0.25) != (true_feat[:, 2].abs() > 0.25)
            )
            joints.append((physical[:, 7:36] - true[:, 7:36]).square())
            velocities.append((physical[:, 36:39] - true[:, 36:39]).square())
            probabilities.append(t.termination(feat).sigmoid())
            labels.append(seq["terminated"][j].float())
            s = t.model.predict_post(
                physical, seq["event_targets"][j], seq["event_mask"][j]
            )
        p, y = torch.stack(probabilities), torch.stack(labels)
        rows[str(h)] = {
            "available": True,
            "batch": 128,
            "root_z_rmse_m": float(torch.stack(errors).mean().sqrt()),
            "anchor_z_feature_rmse_m": float(torch.stack(anchors).mean().sqrt()),
            "anchor_height_terminal_disagreement": float(
                torch.stack(anchor_sign_disagreement).float().mean()
            ),
            "joint_position_rmse_rad": float(torch.stack(joints).mean().sqrt()),
            "root_velocity_rmse_mps": float(torch.stack(velocities).mean().sqrt()),
            "head_predicted_rate": float(p.mean()),
            "true_terminal_rate": float(y.mean()),
            "head_brier": float((p - y).square().mean()),
        }
    result = {
        "completed": True,
        "protocol": "recorded_event_conditions_fixed_real_actions_autoregressive_state",
        "rows": rows,
        "optimizer_updates": 0,
        "warning": "Different horizons sample different surviving windows; not a causal action-gradient or fixed-distribution comparison.",
    }
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "root_tracking.json").write_text(json.dumps(result, indent=2))
    return result
