"""Small pointwise carry fusion, outside native callbacks and random draws."""

import torch


def carry(valid, state, old_state, action, old_action, p, old_p, q, old_q):
    return (
        torch.where(valid[:, None], state, old_state),
        torch.where(valid[:, None], action, old_action),
        torch.where(valid[:, None, None], p, old_p),
        torch.where(valid[:, None, None], q, old_q),
    )


def apply(function, valid, state, old_state, action, old_action, poses, old_poses):
    names = tuple(old_poses[0]), tuple(old_poses[1])
    # The production FK uses identical static names before/after each event.
    if any(set(new) != set(old) for new, old in zip(poses, old_poses)):
        raise ValueError("Carry FK schema changed")
    packed = [
        torch.stack([group[name] for name in keys], 1)
        for keys, new, old in zip(names, poses, old_poses)
        for group in (new, old)
    ]
    state, action, p, q = function(valid, state, old_state, action, old_action, *packed)
    return (
        state,
        action,
        tuple(dict(zip(keys, value.unbind(1))) for keys, value in zip(names, (p, q))),
    )
