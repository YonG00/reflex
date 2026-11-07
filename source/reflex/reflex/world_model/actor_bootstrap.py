"""Exact Actor bootstrap locations; full lambda-target values remain no-grad."""

import torch


def sparse_values(
    t, b, states, previous, phases, modes, truncated, valid, final_observation
):
    h, n = truncated.shape
    cuts = truncated & valid
    # valid becomes false after the first truncation, so at most one cut can
    # contribute per trajectory. Fixed B shapes avoid ragged gathers/recompiles.
    has_cut = cuts.any(0)
    cut = cuts.int().argmax(0)
    env = torch.arange(n, device=truncated.device)
    state = torch.stack(states)[cut + 1, env]
    action = torch.stack(previous)[cut + 1, env]
    phase = torch.stack(phases)[cut + 1, env]
    launched = torch.stack(modes)[cut + 1, env]
    row = {
        key: b[key]
        for key in ("yaw", "dimensions", "mass", "default_q", "default_dq")
        if key in b
    }
    row.update(state=state, previous_action=action, phase=phase, launched=launched)
    poses = t._fk(state)
    observation = t._bootstrap_critic_observation(row, state, action, phase, poses)
    boundary = t.critic(observation).squeeze(-1)
    final = t.critic(final_observation).squeeze(-1)
    values = final.new_zeros(h + 1, n)
    values[cut + 1, env] = torch.where(has_cut, boundary, torch.zeros_like(boundary))
    # The final frame serves both final-step truncation and horizon bootstrap.
    values[-1] = final
    return values
