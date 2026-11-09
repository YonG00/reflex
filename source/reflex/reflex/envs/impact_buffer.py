"""Opt-in PPO impact penalty and diagnostics; no fixed catching posture."""
import json
import os
from pathlib import Path
import torch

from reflex.envs.impact_buffer_core import ImpactBufferConfig, ImpactBufferState

ARM_BODIES = tuple(f'{side}_{part}_link' for side in ('left', 'right')
                   for part in ('elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))


def impact_throw_reset(env, ids, *, hold_init=False):
    """Bookkeeping called IN existing events; never schedules/skips an event."""
    if os.environ.get('BOX_IMPACT_BUFFER', '0') != '1':
        return
    if not hasattr(env, '_impact_hold_init'):
        env._impact_hold_init = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._impact_hold_init[ids] = hold_init
    state = getattr(env, '_impact_buffer_state', None)
    if state is not None:
        if getattr(state, 'log_path', ''):
            # Retain per-throw diagnostics even if a complete 0.5s interaction
            # falls between periodic log snapshots. No events are skipped.
            selected = torch.as_tensor(ids, device=env.device) if not isinstance(ids, slice) else torch.arange(env.num_envs, device=env.device)[ids]
            rows = selected[state.started[selected]].tolist()
            if rows:
                with Path(state.log_path).open('a') as stream:
                    for i in rows:
                        stream.write(json.dumps(dict(event='throw_end', env_id=i,
                            step=int(env.common_step_counter), hold_init=hold_init,
                            contacted=bool(state.contacted[i]), strict=bool(state.strict[i]),
                            first_contact_speed=float(state.first_speed[i]),
                            max_escape_distance=float(state.max_escape[i]),
                            contact_age=float(state.age[i]),
                            episode_step=int(env.episode_length_buf[i]))) + '\n')
        terminated = getattr(env, 'reset_terminated', None)
        state.reset(ids, hold_init=hold_init,
                    terminated=None if terminated is None else terminated[ids])
    sensor = env.scene.sensors.get('impact_box_arm')
    if sensor is not None:
        sensor.clear_history(ids)


def _region_kinematics(env):
    """Nearest points on both elbow->wrist segments and both hand regions."""
    robot, box = env.scene['robot'], env.scene['balloon']
    data = robot.data
    if not hasattr(env, '_impact_body_ids'):
        names = list(robot.body_names)
        env._impact_body_ids = [names.index(name) for name in
            ('left_elbow_link', 'right_elbow_link', 'left_wrist_yaw_link',
             'right_wrist_yaw_link', 'torso_link')]
    le, re, lw, rw, torso = env._impact_body_ids
    bp, bv = box.data.root_pos_w, box.data.root_lin_vel_w
    pos = data.body_pos_w
    com = getattr(data, 'body_com_pos_w', pos)
    lin = getattr(data, 'body_com_lin_vel_w', data.body_lin_vel_w)
    omega = data.body_ang_vel_w
    a, b = pos[:, [le, re]], pos[:, [lw, rw]]
    ab = b - a
    t = ((bp[:, None] - a) * ab).sum(-1) / ab.square().sum(-1).clamp_min(1e-8)
    closest = a + t.clamp(0, 1)[..., None] * ab
    regions = torch.cat((closest, b), dim=1)
    bodies = [le, re, lw, rw]
    vregions = lin[:, bodies] + torch.cross(omega[:, bodies], regions - com[:, bodies], dim=-1)
    delta = bp[:, None] - regions
    distance = delta.norm(dim=-1)
    direction = delta / distance.clamp_min(1e-6)[..., None]
    from isaaclab.utils.math import quat_apply_inverse
    # Direction-dependent box support radius, plus forearm/hand thickness.
    q = box.data.root_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4)
    local_direction = quat_apply_inverse(q, direction.reshape(-1, 3)).reshape(-1, 4, 3)
    support = (local_direction.abs() * env._box_half_extents[:, None]).sum(-1)
    radii = bp.new_tensor([0.035, 0.035, 0.07, 0.07])
    gaps = (distance - support - radii).clamp_min(0)
    selected = gaps.argmin(-1)
    row = torch.arange(env.num_envs, device=bp.device)
    rel = bv[:, None] - vregions
    approach = (-(rel * direction).sum(-1)).clamp_min(0)[row, selected]
    arm_speed = rel.norm(dim=-1)[row, selected]
    tq = data.body_quat_w[:, torso]
    position_b = quat_apply_inverse(tq, bp - pos[:, torso])
    torso_velocity_at_box = lin[:, torso] + torch.cross(
        omega[:, torso], bp - com[:, torso], dim=-1)
    velocity_b = quat_apply_inverse(tq, bv - torso_velocity_at_box)
    return position_b, velocity_b, gaps[row, selected], approach, arm_speed


def impact_buffer_penalty(env, w_pre=0.2, w_damp=0.5, w_escape=1.0,
                          window=0.5, ramp=0.1, pre_range=0.8,
                          speed_scale=8.0, escape_scale=0.6,
                          control_radius=0.65, free_displacement=0.08,
                          contact_force=1.0, log_every=50, log_path=''):
    """Nonpositive sum, with INTERNAL weights; register with RewTerm weight=+1.

    Sensor is strictly box-to-arm filtered, never chest/ground net force.
    First-contact speed is the preceding control-sample relative speed (an
    approximation, not the unobserved exact pre-impulse physics-substep speed).
    Original hold_still/box_on_bed and launch/reset event ordering are untouched.
    """
    if getattr(env, '_wm_ephemeral_view', False):
        raise RuntimeError('impact_buffer_penalty is currently PPO simulator-only')
    cfg = ImpactBufferConfig(w_pre, w_damp, w_escape, window, ramp, pre_range,
        speed_scale, escape_scale, control_radius, free_displacement, contact_force)
    state = getattr(env, '_impact_buffer_state', None)
    if state is None:
        state = env._impact_buffer_state = ImpactBufferState(env.num_envs, env.device)
    if getattr(state, 'log_path', None) != log_path:
        state.log_path = log_path
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    step = int(env.common_step_counter)
    if state.last_step == step:
        return state.cached['total']
    state.suppressed.copy_(getattr(env, '_impact_hold_init', state.suppressed))
    sensor = env.scene['impact_box_arm']
    _ = sensor.data  # force eager/lazy buffer refresh before reading pair history
    forces = sensor.pair_force_history
    contact = (forces.norm(dim=-1) > cfg.contact_force).flatten(1).any(-1)
    position, velocity, gap, approach, arm_speed = _region_kinematics(env)
    from reflex.envs.base.chest_v5_terms import compute_strict_components
    strict = compute_strict_components(env)['strict_all']
    result = state.update(step=step, dt=env.step_dt, launched=env._box_launched,
        contact=contact, position_b=position, velocity_b=velocity, gap=gap,
        approach=approach, arm_speed=arm_speed, strict_now=strict, cfg=cfg)
    if log_every > 0 and step % log_every == 0:
        def mean(value, mask):
            return float((value * mask).sum().detach().cpu() / mask.sum().clamp_min(1).cpu())
        counts = {k: float(v.detach().cpu()) for k, v in state.counts.items()}
        metrics = dict(step=step, **counts,
            pre_approach_speed=mean(approach, result['pre_gate']),
            first_contact_speed=mean(state.first_speed, state.contacted),
            post_relative_speed=mean(result['post_speed'], result['post_gate']),
            max_escape_distance=float(state.max_escape_seen.detach().cpu()),
            mean_throw_max_escape=mean(state.max_escape, state.contacted),
            pre_active=int(result['pre_gate'].sum()), post_active=int(result['post_gate'].sum()),
            penalty_pre=float(result['parts'][:, 0].mean().detach().cpu()),
            penalty_damp=float(result['parts'][:, 1].mean().detach().cpu()),
            penalty_escape=float(result['parts'][:, 2].mean().detach().cpu()),
            penalty_total=float(result['total'].mean().detach().cpu()),
            first_arm_contact_rate=counts['contacts'] / max(counts['throws'], 1),
            strict_catch_rate=counts['strict_success'] / max(counts['throws'], 1))
        env.extras.setdefault('log', {}).update({f'Impact/{k}': v for k, v in metrics.items() if k != 'step'})
        print('[impact_buffer] ' + json.dumps(metrics), flush=True)
        if log_path:
            target = Path(log_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('a') as stream:
                stream.write(json.dumps(metrics) + '\n')
    return result['total']
