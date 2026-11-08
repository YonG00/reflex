"""Opt-in execution changes; native callbacks, clocks and RNG order stay intact."""

import copy
import functools
import types

import torch

from .writes import PlannedAsset, WritePlan


class PoseTrackedAsset(PlannedAsset):
    """Track issued pose writes on the host, never inspect CUDA mask values."""

    def __init__(self, *args, tracker, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = tracker

    def write_root_pose_to_sim(self, pose, env_ids=None):
        if self.robot:
            self.tracker.robot_pose_write_issued = True
        return super().write_root_pose_to_sim(pose, env_ids)

    def write_joint_state_to_sim(
        self, positions, velocities, joint_ids=None, env_ids=None
    ):
        if self.robot:
            self.tracker.robot_pose_write_issued = True
        return super().write_joint_state_to_sim(
            positions, velocities, joint_ids, env_ids
        )


def configure(
    scheduler, track_pose=False, compact_pools=False, collapse_full_subsets=False
):
    if not hasattr(scheduler, "_v30_original_step"):
        scheduler._v30_original_step = scheduler.step
    if not hasattr(scheduler, "_v30_original_invoke"):
        scheduler._v30_original_invoke = scheduler._invoke
    scheduler._invoke = (
        types.MethodType(full_subset_invoke, scheduler)
        if collapse_full_subsets
        else scheduler._v30_original_invoke
    )
    if track_pose:
        scheduler.event_asset_factory = functools.partial(
            PoseTrackedAsset, tracker=scheduler
        )
    elif hasattr(scheduler, "event_asset_factory"):
        del scheduler.event_asset_factory
    scheduler.robot_pose_write_issued = False
    scheduler.step = (
        types.MethodType(compact_pool_step, scheduler)
        if compact_pools
        else scheduler._v30_original_step
    )


def full_subset_invoke(
    self, term, b, physical, poses, values, episode, common, plan, selected, subset
):
    """Elide identity gathers/copies for a full ascending nonzero selection.

    Native step constructs selected with nonzero(), hence length N proves
    selected == arange(N). Callbacks marked ignores_ids still receive None,
    exactly as the original subset path. RNG and every callback are retained.
    """
    if subset and len(physical) and len(selected) == len(physical):
        view = self._view(b, physical, poses, values, episode, common, plan)
        term.function(view, None, **term.params)
        result = {name: getattr(view, name) for name in values}
        if hasattr(view, "_box_posture_ids"):
            self.static["_box_posture_ids"] = copy.deepcopy(view._box_posture_ids)
        return result
    return self._v30_original_invoke(
        term, b, physical, poses, values, episode, common, plan, selected, subset
    )


def compact_pool_step(
    self,
    b,
    physical,
    poses,
    state,
    pools,
    episode,
    common,
    shared_world=False,
    isolated_rng=True,
):
    """Functional timer update without cloning a whole pool twice.

    Pool topology and foreign-world triggers are retained. Random draws use the
    exact same due indices/count/order as NativeEventScheduler.step.
    """
    if isolated_rng:
        with self.random_stream():
            return self.step(
                b, physical, poses, state, pools, episode, common, shared_world, False
            )
    values, timers = self.codec.decode(state)
    n = len(physical)
    plan = getattr(self, "event_plan_factory", WritePlan)(physical)
    updated_pools = [None] * len(self.pool_terms)
    for i, term in enumerate(self.terms):
        lower, upper = term.interval
        if i in self.pool_index:
            k = self.pool_index[i]
            source = pools[0, k : k + 1] if shared_world else pools[:, k]
            # Subtraction owns new storage, so resetting due entries needs no clone.
            pool = source - self.task.dt
            due = pool < 1e-6
            ids = due.nonzero(as_tuple=True)
            if len(ids[0]):
                pool[ids] = (
                    torch.rand(len(ids[0]), device=physical.device) * (upper - lower)
                    + lower
                )
            fire = due.any(-1)
            if shared_world:
                updated_pools[k] = pool.expand(n, -1)
                timers[:, i] = pool[0, b["event_agent"]]
                fire = fire.expand(n)
            else:
                updated_pools[k] = pool
                timers[:, i] = pool[
                    torch.arange(n, device=physical.device), b["event_agent"]
                ]
            selected = fire.nonzero().flatten()
            values = self._invoke(
                term,
                b,
                physical,
                poses,
                values,
                episode,
                common,
                plan,
                selected,
                term.ignores_ids,
            )
        else:
            remaining = timers[:, i] - self.task.dt
            selected = (remaining < 1e-6).nonzero().flatten()
            remaining = remaining.clone()
            if len(selected):
                remaining[selected] = (
                    torch.rand(len(selected), device=physical.device) * (upper - lower)
                    + lower
                )
            timers[:, i] = remaining
            values = self._invoke(
                term,
                b,
                physical,
                poses,
                values,
                episode,
                common,
                plan,
                selected,
                term.ignores_ids,
            )
    if not updated_pools:
        next_pools = pools.clone()
    elif len(updated_pools) == 1 and not shared_world:
        next_pools = updated_pools[0].unsqueeze(1)
    else:
        next_pools = torch.stack(updated_pools, dim=1)
    return plan, self.codec.encode(values, timers), next_pools
