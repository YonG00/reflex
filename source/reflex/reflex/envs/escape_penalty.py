"""Severe escape failure; no pre-contact or continuous velocity penalties."""
import json
from pathlib import Path
import torch
from reflex.envs.escape_penalty_core import EscapeConfig, EscapeState


def _state(env):
    if not hasattr(env, '_escape_state'):
        env._escape_state = EscapeState(env.num_envs, env.device)
    return env._escape_state


def _clear_contact(env, ids):
    sensor = env.scene.sensors.get('escape_box_robot')
    if sensor is not None:
        sensor.clear_history(ids)


def escape_throw_reset(env, ids, hold_init=False):
    state = _state(env)
    if ids is None:
        ids = slice(None)
    terminated = getattr(env, 'reset_terminated', None)
    if getattr(state, 'log_path', ''):
        indices = torch.arange(env.num_envs, device=env.device)[ids]
        rows = indices[state.started[indices]].tolist()
        if rows:
            with Path(state.log_path).open('a') as stream:
                for i in rows:
                    stream.write(json.dumps(dict(event='throw_end', step=int(env.common_step_counter),
                        env_id=i, throw_id=int(state.throw_id[i]), escaped=bool(state.escaped[i]),
                        contacted=bool(state.ever_contacted[i]), caught=bool(state.caught[i]),
                        episode_step=int(env.episode_length_buf[i]),
                        terminated=False if terminated is None else bool(terminated[i]))) + '\n')
    state.throw_reset(ids, hold_init=hold_init,
                      terminated=None if terminated is None else terminated[ids])
    _clear_contact(env, ids)


def torso_relative_kinematics(env):
    from isaaclab.utils.math import quat_apply_inverse
    data, box = env.scene['robot'].data, env.scene['balloon'].data
    if not hasattr(env, '_escape_torso_id'):
        env._escape_torso_id = list(env.scene['robot'].body_names).index('torso_link')
    i = env._escape_torso_id
    origin = data.body_pos_w[:, i]
    com = getattr(data, 'body_com_pos_w', data.body_pos_w)[:, i]
    lin = getattr(data, 'body_com_lin_vel_w', data.body_lin_vel_w)[:, i]
    at_box = lin + torch.cross(data.body_ang_vel_w[:, i], box.root_pos_w - com, dim=-1)
    q = data.body_quat_w[:, i]
    return (quat_apply_inverse(q, box.root_pos_w - origin),
            quat_apply_inverse(q, box.root_lin_vel_w - at_box))


def escape_penalty(env, amount=100., distance=1., radial_speed=1., persistence=.15,
                   contact_force=1., log_every=50, log_path=''):
    if getattr(env, '_wm_ephemeral_view', False):
        raise RuntimeError('escape_penalty currently requires the PPO simulator')
    state = _state(env)
    step = int(env.common_step_counter)
    if state.last_step == step:
        return state.cached['event'] / env.step_dt
    cfg = EscapeConfig(amount, distance, radial_speed, persistence, contact_force)
    if getattr(state, 'log_path', None) != log_path:
        state.log_path = log_path
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    sensor = env.scene['escape_box_robot']
    _ = sensor.data
    contact = (sensor.pair_force_history.norm(dim=-1) > cfg.contact_force).flatten(1).any(-1)
    position, velocity = torso_relative_kinematics(env)
    from reflex.envs.base.chest_v5_terms import compute_strict_components
    result = state.update(step=step, dt=env.step_dt, launched=env._box_launched,
        contact=contact, position_b=position, velocity_b=velocity,
        strict_now=compute_strict_components(env)['strict_all'], cfg=cfg)
    count = state.counts
    denom = count['throws'].clamp_min(1)
    metrics = dict(escape_penalty=result['event'].mean(),
        escape_rate=count['escapes']/denom, contact_rate=count['contacts']/denom,
        catch_success_rate=count['catches']/denom)
    env.extras.setdefault('log', {}).update({f'Escape/{k}': v.detach() for k,v in metrics.items()})
    if log_path and bool(result['new_escape'].any()):
        # Never lose a one-shot event between the periodic snapshots.
        with Path(log_path).open('a') as stream:
            for i in result['new_escape'].nonzero(as_tuple=False).flatten().tolist():
                stream.write(json.dumps(dict(event='escape', step=step, env_id=i, throw_id=int(state.throw_id[i]),
                    episode_step=int(env.episode_length_buf[i]), escape_penalty=-cfg.amount, radius=float(result['radius'][i]),
                    radial_speed=float(result['radial_speed'][i]), dwell=float(result['dwell'][i])))+'\n')
    if log_every > 0 and step % log_every == 0:
        row = dict(step=step, **{k: float(v.detach().cpu()) for k,v in metrics.items()},
                   **{k: float(v.detach().cpu()) for k,v in count.items()})
        print('[escape_penalty] '+json.dumps(row), flush=True)
        if log_path:
            with Path(log_path).open('a') as stream:
                stream.write(json.dumps(row)+'\n')
    # RewardManager multiplies every term by dt. Compensate here so the
    # actual task-reward impulse is -amount, NOT -amount*dt. Outer weight=+1.
    return result['event'] / env.step_dt
