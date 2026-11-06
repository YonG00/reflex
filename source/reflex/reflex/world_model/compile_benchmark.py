"""Read-only real-replay equivalence and full BPTT timing, no optimizer steps."""

import time
import torch
from .trainer import freeze
from .joint import returns


def check(t, repeats=3):
    def sync():
        if str(t.device).startswith("cuda"):
            torch.cuda.synchronize()

    def run(b):
        with freeze(t.model, t.termination, t.prior._disc, t.critic):
            v = t.imagine(start_batch=b)
            loss = -returns(v, t.critic, t.cfg.gamma).mean()
            if t.cfg.actor_horizon_scale:
                loss = loss / t.cfg.horizon
            g = torch.autograd.grad(loss, tuple(t.actor.parameters()))
        return {
            k: v[k].detach()
            for k in ("obs", "critic_obs", "reward", "done", "truncated")
        }, tuple(x.detach() for x in g)

    old = (t._transition, t._observations, t._task_step, t._termination_fn)
    fused = t.transition
    t._task_step = torch.compile(
        t.task.evaluate_step, fullgraph=True, backend=t.cfg.compile_backend
    )
    t._termination_fn = torch.compile(
        t.termination, fullgraph=True, backend=t.cfg.compile_backend
    )
    observation = torch.compile(
        t.imagined_observations, fullgraph=True, backend=t.cfg.compile_backend
    )
    result = {
        "optimizer_updates": 0,
        "batch": t.cfg.actor_batch,
        "horizon": t.cfg.horizon,
    }
    devices = (
        [torch.device(t.device).index or 0] if str(t.device).startswith("cuda") else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(114)
            b = t.sample(starts=True, n=8)
            # Initialize original reward constants/log-once caches outside capture.
            cores = t._dynamics, t._fk
            t._dynamics, t._fk = t.model, t.bridge.fk
            try:
                run(b)
            finally:
                t._dynamics, t._fk = cores
            print("FULLGRAPH_CAPTURE_START", flush=True)
            t._transition = torch.compile(t.transition, fullgraph=True, backend="eager")
            t._observations = torch.compile(
                t.imagined_observations, fullgraph=True, backend="eager"
            )
            run(b)
            print("FULLGRAPH_CAPTURE_PASSED", flush=True)
            t._transition, t._observations = None, t.imagined_observations
            eager, eg = run(b)
            print("BASELINE_EQUIVALENCE_READY", flush=True)
            t._transition, t._observations = fused, observation
            actual, ag = run(b)
            for key in eager:
                torch.testing.assert_close(
                    eager[key],
                    actual[key],
                    atol=2e-4,
                    rtol=3e-3,
                    msg=lambda message: key + ": " + message,
                )
            for x, y in zip(eg, ag):
                torch.testing.assert_close(x, y, atol=2e-5, rtol=3e-3)
            result["equivalence_passed"] = True
            result["actor_gradient_max_abs_diff"] = max(
                float((x - y).abs().max()) for x, y in zip(eg, ag)
            )
            print("EXPANDED_COMPILE_EQUIVALENCE_PASSED", flush=True)
            del eager, eg, actual, ag
            b = t.sample(starts=True, n=t.cfg.actor_batch or t.cfg.batch)
            for name, functions in [
                ("wm_fk", (None, t.imagined_observations)),
                ("expanded", (fused, observation)),
            ]:
                t._transition, t._observations = functions
                run(b)  # warm compile separately from steady-state timing
                sync()
                times = []
                for _ in range(repeats):
                    start = time.monotonic()
                    run(b)
                    sync()
                    times.append(time.monotonic() - start)
                result[name + "_bptt_seconds"] = times
                print(name, times, flush=True)
    finally:
        t._transition, t._observations, t._task_step, t._termination_fn = old
    return result
