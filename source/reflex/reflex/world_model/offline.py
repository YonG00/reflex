"""Fit/evaluate ONLY WM on frozen real replay. No simulator, Actor, Critic or AMP.

Example: python -m reflex.world_model.offline --checkpoint ... --urdf ... --output ...
The recorded real actions drive every prediction step. Comparisons share replay,
initial parameters/Adam, random starts, and fixed held-out evaluation windows.
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from torch import nn
from .models import WorldModel, StateWorldModel
from .physics import Kinematics, state_error
from .observation import OracleObservation
from .replay import Replay
from .timing import timestamp
from reflex.paths import REPO_ROOT


def observation(bridge, b, state):
    return bridge(
        state,
        b["previous_action"],
        b["phase"],
        b["yaw"],
        b["launched"],
        b["dimensions"],
        b["mass"],
        b.get("default_q"),
        b.get("default_dq"),
    )


def rollout(model, bridge, seq, horizon):
    if model.state_only:
        raise TypeError(
            "State-only WM has no reward/memory/done outputs; use predict_states"
        )
    state, memory = seq["state"][0], seq["memory"][0]
    states = []
    memories = []
    rewards = []
    dones = []
    for t in range(horizon):
        b = {k: v[t] for k, v in seq.items()}
        state, memory, reward, done = model(
            state,
            b["target"],
            b["context"],
            memory,
            b["phase"],
            b["next_blend"],
            observation(bridge, b, state),
            b["action"],
        )
        states.append(state)
        memories.append(memory)
        rewards.append(reward)
        dones.append(done)
    return (
        torch.stack(states),
        torch.stack(memories),
        torch.stack(rewards),
        torch.stack(dones),
    )


def predict_states(model, seq, horizon):
    state = seq["state"][0]
    states = []
    for t in range(horizon):
        state = model(state, seq["target"][t], seq["context"][t])
        states.append(state)
    return torch.stack(states)


def window_loss(model, bridge, seq, horizon, state_weight=1.0):
    if model.state_only:
        states = predict_states(model, seq, horizon)
        return (
            state_weight
            * state_error(states, seq["next_state"][:horizon]).square().mean()
        )
    states, memories, rewards, dones = rollout(model, bridge, seq, horizon)
    sl = state_error(states, seq["next_state"][:horizon]).square().mean()
    rl = (
        ((rewards - seq["reward_terms"][:horizon]) / model.reward_norm.std)
        .square()
        .mean()
    )
    ml = (memories - seq["next_memory"][:horizon]).square().mean()
    dl = nn.functional.binary_cross_entropy_with_logits(
        dones, seq["terminated"][:horizon].float()
    )
    return state_weight * sl + rl + 0.1 * ml + dl


@torch.no_grad()
def evaluate(model, bridge, batches):
    result = {}
    for horizon, seq in batches.items():
        if model.state_only:
            states = predict_states(model, seq, horizon)
            rewards = None
        else:
            states, _, rewards, _ = rollout(model, bridge, seq, horizon)
        pred, true = states[-1], seq["next_state"][-1]
        error = state_error(pred, true)
        values = {
            "state_rmse": float(error.square().mean(-1).sqrt().mean()),
            "state_nse": float(error.square().mean()),
            "robot_nse": float(error[:, :70].square().mean()),
            "box_nse": float(error[:, 70:].square().mean()),
            "box_position_m": float(
                (pred[:, 71:74] - true[:, 71:74]).square().mean().sqrt()
            ),
            "root_position_m": float(
                (pred[:, :3] - true[:, :3]).square().mean().sqrt()
            ),
            "joint_position_rad": float(
                (pred[:, 7:36] - true[:, 7:36]).square().mean().sqrt()
            ),
            "box_angular_velocity_radps": float(
                (pred[:, 81:84] - true[:, 81:84]).square().mean().sqrt()
            ),
            "start_stage_counts": torch.bincount(seq["stage"][0], minlength=4).tolist(),
        }
        for name, sl in [
            ("root_linear_velocity_mps", slice(36, 39)),
            ("root_angular_velocity_radps", slice(39, 42)),
            ("joint_velocity_radps", slice(42, 71)),
            ("box_linear_velocity_mps", slice(78, 81)),
        ]:
            values[name] = float((pred[:, sl] - true[:, sl]).square().mean().sqrt())
        for name, sl in [
            ("root_rotation_rad", slice(3, 7)),
            ("box_rotation_rad", slice(74, 78)),
        ]:
            dot = (pred[:, sl] * true[:, sl]).sum(-1).abs().clamp(0, 1)
            values[name] = float((2 * torch.acos(dot)).square().mean().sqrt())
        values["box_position_3d_rmse_m"] = float(
            (pred[:, 71:74] - true[:, 71:74]).square().sum(-1).mean().sqrt()
        )
        if rewards is not None:
            values["reward_rmse"] = float(
                ((rewards[-1] - seq["reward_terms"][-1]) / model.reward_norm.std)
                .square()
                .mean()
                .sqrt()
            )
        result[str(horizon)] = values
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--updates", type=int, default=64)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--objective", choices=("multitask", "state"), default="multitask")
    p.add_argument(
        "--hidden_dims", help="WM hidden widths, e.g. 2048,2048,2048; requires --fresh"
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Fresh WM/Adam; reuse only frozen replay and training-only normalization",
    )
    p.add_argument("--lr", type=float)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--activation", choices=("elu", "silu"), default="elu")
    p.add_argument(
        "--fixed_minibatch",
        action="store_true",
        help="Overfit one fixed training batch to diagnose representational/optimization errors",
    )
    p.add_argument(
        "--cases", default="4:1,8:1,8:82", help="window:state_loss_weight pairs"
    )
    args = p.parse_args()
    if args.objective == "state" and not args.fresh:
        p.error(
            "Pure-state WM uses a different schema; old multitask checkpoints cannot resume it. Use --fresh"
        )
    if (args.hidden_dims or args.activation != "elu") and not args.fresh:
        p.error("Architecture/activation overrides require --fresh")
    torch.set_num_threads(1)
    cases = [
        (int(s.split(":")[0]), float(s.split(":")[1])) for s in args.cases.split(",")
    ]
    if any(h < 1 or w <= 0 for h, w in cases):
        p.error("Positive windows and weights required")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    c, conf, sd = ck["contract"], ck["config"], ck["bridge"]
    if hashlib.sha256(Path(args.urdf).read_bytes()).hexdigest() != c["urdf_sha256"]:
        raise ValueError("URDF checksum differs from data contract")
    fk = Kinematics(args.urdf, c["joint_names"])
    bridge = OracleObservation(
        fk,
        sd["motion_q"],
        sd["motion_dq"],
        sd["anchor_pos"],
        sd["anchor_quat"],
        sd["default_q"],
        sd["default_dq"],
        "torso_link",
        sd["reference_arm_columns"],
        sd["reference_down"],
    ).to(args.device)
    bridge.load_state_dict(sd)
    replay = Replay(conf["capacity_steps"], args.device)
    replay.load_state_dict(ck["replay"])
    torch.manual_seed(args.seed)
    hidden_layers = (
        [int(x) for x in args.hidden_dims.split(",")] if args.hidden_dims else None
    )
    model = (
        StateWorldModel(
            c["context_dim"],
            hidden_layers=hidden_layers or [2048] * 3,
            com_indices=c["com_indices"],
            activation=nn.SiLU if args.activation == "silu" else nn.ELU,
        )
        if args.objective == "state"
        else WorldModel(
            c["context_dim"],
            len(c["memory"]),
            len(ck["reward_names"]),
            hidden=conf["hidden"],
            com_indices=c["com_indices"],
            hidden_layers=hidden_layers,
            activation=nn.SiLU if args.activation == "silu" else nn.ELU,
        )
    ).to(args.device)
    if args.fresh:
        # The same training-only normalization pool is used across all four
        # designs; overlapping physical features get identical scales.
        torch.manual_seed(args.seed + 11)
        norm = {k: v[0] for k, v in replay.sample(2048, 1).items()}
        if model.state_only:
            model.inputs.fit(
                model.features(norm["state"], norm["target"], norm["context"])
            )
        else:
            model.inputs.fit(
                model.features(
                    norm["state"],
                    norm["target"],
                    norm["context"],
                    norm["memory"],
                    norm["phase"],
                    norm["next_blend"],
                    observation(bridge, norm, norm["state"]),
                    norm["action"],
                )
            )
            model.reward_norm.fit(norm["reward_terms"])
    else:
        model.load_state_dict(ck["model"])
    initial = copy.deepcopy(model.state_dict())
    torch.manual_seed(args.seed)
    batches = {h: replay.sample(256, h, validation=True) for h in (1, 4, 8, 16, 32)}
    result = {
        "source": args.checkpoint,
        "source_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "code_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "args": vars(args),
        "source_iteration": ck["iteration"],
        "actor_called": False,
        "warm_start_actor": False,
        "stage_balanced": False,
        "simulator_transitions": 0,
        "model_schema": "state_v2" if model.state_only else "legacy_multitask_v1",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "baseline": evaluate(model, bridge, batches),
        "cases": [],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_horizon = max(h for h, _ in cases)
    for horizon, weight in cases:
        model.load_state_dict(initial)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr or conf["lr"])
        if not args.fresh:
            opt.load_state_dict(copy.deepcopy(ck["model_opt"]))
            if args.lr is not None:
                for group in opt.param_groups:
                    group["lr"] = args.lr
        # Each case draws identical start states/windows. Only the prefix used
        # for loss differs, so long-window eligibility cannot change sampling.
        torch.manual_seed(args.seed + 1)
        fixed_seq = (
            replay.sample(args.batch, train_horizon) if args.fixed_minibatch else None
        )
        torch.manual_seed(args.seed + 2)
        train_probe = (
            fixed_seq if fixed_seq is not None else replay.sample(256, train_horizon)
        )
        train_before = evaluate(
            model, bridge, {horizon: {k: v[:horizon] for k, v in train_probe.items()}}
        )
        torch.manual_seed(args.seed + 1)
        begin = timestamp(args.device)
        for update in range(args.updates):
            seq = (
                fixed_seq
                if fixed_seq is not None
                else replay.sample(args.batch, train_horizon)
            )
            loss = window_loss(model, bridge, seq, horizon, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            opt.step()
            if (update + 1) % 16 == 0:
                print(
                    json.dumps(
                        {
                            "window": horizon,
                            "state_weight": weight,
                            "update": update + 1,
                            "loss": float(loss.detach()),
                        }
                    ),
                    flush=True,
                )
        row = {
            "window": horizon,
            "state_weight": weight,
            "optimizer_updates": args.updates,
            "model_transition_evaluations": args.updates * args.batch * horizon,
            "seconds": timestamp(args.device) - begin,
            "evaluation": evaluate(model, bridge, batches),
            "training_probe_before": train_before,
            "training_probe_after": evaluate(
                model,
                bridge,
                {horizon: {k: v[:horizon] for k, v in train_probe.items()}},
            ),
        }
        result["cases"].append(row)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "source": args.checkpoint,
                "case": row,
                "contract": c,
                "source_training_config": conf,
                "offline_fit_args": vars(args),
            },
            output / f"wm_window{horizon}_weight{weight:g}.pt",
        )
        (output / "comparison.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
