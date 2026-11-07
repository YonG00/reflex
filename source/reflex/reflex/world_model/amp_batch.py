"""Build pure AMP features after the sequential physical/event trajectory."""

import torch

from .physics import amp_features

KEYS = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


def feature_pairs(rows):
    # Rows hold immutable tensors from the exact pre-control/Physical clocks.
    # No FK recomputation, history update, detach or event reordering occurs.
    states = torch.stack([s for before, after, _, _ in rows for s in (before, after)])
    positions = {
        key: torch.stack(
            [p[key] for _, _, before, after in rows for p in (before, after)]
        ).flatten(0, 1)
        for key in KEYS
    }
    features = amp_features(states.flatten(0, 1), None, (positions, {}))
    features = features.reshape(len(rows), 2, states.shape[1], -1)
    return features[:, 0], features[:, 1]
