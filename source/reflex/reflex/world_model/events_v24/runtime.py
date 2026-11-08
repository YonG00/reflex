"""Run the SAME native event callbacks against a state-backed write recorder.

Only commands/controller state are returned. This module never returns a
replacement physical state to the imagined plant.
"""

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ..task import Scene, primitive
from .state import EventStateCodec
from .fast_view import build as minimal_view
from .writes import PlannedAsset, WritePlan


@dataclass
class EventTerm:
    name: str
    function: object
    params: dict
    interval: tuple
    global_time: bool
    ignores_ids: bool


SUPPORTED = {
    "push_by_setting_velocity": False,
    "v38_walking_hover_and_launch": True,
    "hold_init_apply": True,
    "first_throw_realign": True,
    "v38_cradle_perturb": True,
}


class NativeEventScheduler:
    def __init__(self, env, task, seed=24001):
        self.task = task
        manager = env.event_manager
        self.terms = []
        for name, cfg in zip(
            manager._mode_term_names["interval"], manager._mode_term_cfgs["interval"]
        ):
            fn = cfg.func.__name__
            if cfg.is_global_time:
                raise ValueError(
                    f"V24 global-time callback needs explicit verification: {name}"
                )
            if fn not in SUPPORTED:
                raise ValueError(
                    f"V24 has no verified adapter for active event {name}: {fn}"
                )
            self.terms.append(
                EventTerm(
                    name,
                    cfg.func,
                    cfg.params,
                    tuple(cfg.interval_range_s),
                    cfg.is_global_time,
                    SUPPORTED[fn],
                )
            )
        self.codec = EventStateCodec(env, self.terms)
        self.source_count = env.num_envs
        self.pool_terms = [
            i
            for i, t in enumerate(self.terms)
            if t.global_time or (t.ignores_ids and t.interval[1] > 0)
        ]
        self.pool_index = {index: j for j, index in enumerate(self.pool_terms)}
        self.static = {
            name: copy.deepcopy(getattr(env, name))
            for name in ("_box_posture_ids", "_v36_lim")
            if hasattr(env, name)
        }
        self.device = str(env.device)
        generator = torch.Generator(device=env.device)
        generator.manual_seed(seed)
        self.rng = generator.get_state()

    def specification(self):
        import hashlib
        import sys
        from pathlib import Path

        sources = {}
        constants = {}
        for term in self.terms:
            module = sys.modules[term.function.__module__]
            path = Path(module.__file__)
            sources[term.function.__module__] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            values = {}
            for name, value in vars(module).items():
                if (
                    name.isupper()
                    and not name.startswith("_")
                    and isinstance(value, (bool, int, float, str, tuple, list, dict))
                ):
                    try:
                        values[name] = primitive(value)
                    except TypeError:
                        pass
            constants[term.function.__module__] = values
        for path in (
            Path(__file__),
            Path(__file__).with_name("state.py"),
            Path(__file__).with_name("writes.py"),
            Path(__file__).parents[2] / "event_math.py",
        ):
            if path.exists():
                sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "source_sha256": sources,
            "constants": constants,
            "schema": 1,
            "state": self.codec.specification(),
            "source_world_envs": self.source_count,
            "terms": [
                {
                    "name": t.name,
                    "function": t.function.__module__ + "." + t.function.__name__,
                    "parameters": primitive(t.params),
                    "interval": list(t.interval),
                    "global_time": t.global_time,
                    "ignores_ids": t.ignores_ids,
                }
                for t in self.terms
            ],
        }

    def snapshot(self, env):
        timers = env.event_manager._interval_term_time_left
        state = self.codec.capture(env, timers)
        # One per vector step, not N copies in Replay.
        pools = (
            torch.stack([timers[i].expand(env.num_envs) for i in self.pool_terms])
            if self.pool_terms
            else state.new_empty(0, env.num_envs)
        )
        return {
            "event_state": state,
            "event_pool": pools.clone(),
            "event_origins": env.scene.env_origins.clone(),
            "event_agent": torch.arange(env.num_envs, device=env.device),
        }

    @contextmanager
    def random_stream(self):
        devices = (
            [torch.device(self.device).index or 0]
            if self.device.startswith("cuda")
            else []
        )
        with torch.random.fork_rng(devices=devices):
            if devices:
                torch.cuda.set_rng_state(self.rng, devices[0])
            else:
                torch.set_rng_state(self.rng)
            try:
                yield
            finally:
                self.rng = (
                    torch.cuda.get_rng_state(devices[0])
                    if devices
                    else torch.get_rng_state()
                )

    def _view(self, b, physical, poses, values, episode, common, plan):
        n = len(physical)
        # Stage native commands for subsequent callbacks in the SAME interval.
        # This shadow is never returned as the WM prediction.
        physical = torch.where(plan.mask.clone(), plan.targets, physical)
        hist = b["task_history"]
        view = self.task._view(
            b,
            physical,
            physical,
            b["previous_action"],
            b["previous_action"],
            b["phase"],
            hist,
            episode,
            1,
            poses,
            poses,
            False,
        )
        origins = b["event_origins"]
        view.scene.env_origins = origins
        robot, box = view.scene["robot"], view.scene["balloon"]
        for asset in (robot, box):
            d = asset.data
            for key, value in vars(d).items():
                if torch.is_tensor(value):
                    setattr(d, key, value.clone())
            d.root_pos_w = d.root_pos_w + origins
            if hasattr(d, "body_pos_w"):
                d.body_pos_w = d.body_pos_w + origins[:, None, :]
        view.scene = Scene(
            robot=PlannedAsset(
                robot.data, robot.body_names, robot.joint_names, plan, origins, True
            ),
            balloon=PlannedAsset(
                box.data, box.body_names, box.joint_names, plan, origins, False
            ),
        )
        view.scene.env_origins = origins
        view.num_envs, view.common_step_counter = n, common
        for name, value in self.static.items():
            setattr(view, name, copy.deepcopy(value))
        for name, value in values.items():
            setattr(view, name, value.clone())
        # Native cached geometry already describes each spawned box.
        view._box_half_extents = b["dimensions"] / 2
        return view

    def _invoke(
        self, term, b, physical, poses, values, episode, common, plan, selected, subset
    ):
        if not len(selected):
            return values
        if subset:
            if (
                getattr(self, "compact_inputs", False)
                and getattr(self._view, "__func__", None) is minimal_view
            ):
                # fast_view.build reads exactly these fields and ignores poses.
                # Do not gather full replay context or every body pose only to
                # discard them immediately in the minimal event facade.
                bb = {
                    k: b[k][selected]
                    for k in ("event_origins", "task_history", "dimensions")
                }
                pp = poses
            else:
                bb = {
                    k: (
                        v[selected]
                        if torch.is_tensor(v) and len(v) == len(physical)
                        else v
                    )
                    for k, v in b.items()
                }
                pp = tuple(
                    {k: v[selected] for k, v in group.items()} for group in poses
                )
            vv = {k: v[selected] for k, v in values.items()}
            child_plan = plan.select(selected)
            view = self._view(
                bb,
                physical[selected],
                pp,
                vv,
                episode[selected],
                common[selected],
                child_plan,
            )
            term.function(view, None, **term.params)
            for name in values:
                values[name] = values[name].index_copy(0, selected, getattr(view, name))
            plan.merge(selected, child_plan)
        else:
            view = self._view(b, physical, poses, values, episode, common, plan)
            term.function(view, selected, **term.params)
            values = {name: getattr(view, name) for name in values}
        if hasattr(view, "_box_posture_ids"):
            self.static["_box_posture_ids"] = copy.deepcopy(view._box_posture_ids)
        return values

    def step(
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
        if isolated_rng:
            with self.random_stream():
                return self.step(
                    b,
                    physical,
                    poses,
                    state,
                    pools,
                    episode,
                    common,
                    shared_world,
                    False,
                )
        values, timers = self.codec.decode(state)
        n = len(physical)
        plan = getattr(self, "event_plan_factory", WritePlan)(physical)
        next_pools = pools.clone()
        for i, term in enumerate(self.terms):
            lower, upper = term.interval
            if i in self.pool_index:
                k = self.pool_index[i]
                # shared_world is used only for native equivalence tests.
                pool = next_pools[0, k : k + 1] if shared_world else next_pools[:, k]
                pool = pool - self.task.dt
                due = pool < 1e-6
                ids = due.nonzero(as_tuple=True)
                if len(ids[0]):
                    pool = pool.clone()
                    pool[ids] = (
                        torch.rand(len(ids[0]), device=physical.device)
                        * (upper - lower)
                        + lower
                    )
                fire = due.any(-1)
                if shared_world:
                    next_pools[:, k] = pool.expand(n, -1)
                    timers[:, i] = pool[0, b["event_agent"]]
                    fire = fire.expand(n)
                else:
                    next_pools[:, k] = pool
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
                        torch.rand(len(selected), device=physical.device)
                        * (upper - lower)
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
        return plan, self.codec.encode(values, timers), next_pools
