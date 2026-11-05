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
import torch
from torch import nn
from .models import WorldModel
from .physics import Kinematics, state_error
from .observation import OracleObservation
from .replay import Replay
from .timing import timestamp


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


def window_loss(model, bridge, seq, horizon, state_weight=1.0):
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
        states, _, rewards, _ = rollout(model, bridge, seq, horizon)
        pred, true = states[-1], seq["next_state"][-1]
        values = {
            "state_rmse": float(
                state_error(pred, true).square().mean(-1).sqrt().mean()
            ),
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
            "reward_rmse": float(
                ((rewards[-1] - seq["reward_terms"][-1]) / model.reward_norm.std)
                .square()
                .mean()
                .sqrt()
            ),
            "start_stage_counts": torch.bincount(seq["stage"][0], minlength=4).tolist(),
        }
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
    p.add_argument(
        "--cases", default="4:1,8:1,8:82", help="window:state_loss_weight pairs"
    )
    args = p.parse_args()
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
    model = WorldModel(
        c["context_dim"],
        len(c["memory"]),
        len(ck["reward_names"]),
        hidden=conf["hidden"],
        com_indices=c["com_indices"],
    ).to(args.device)
    model.load_state_dict(ck["model"])
    torch.manual_seed(args.seed)
    batches = {h: replay.sample(256, h, validation=True) for h in (1, 4, 8, 16, 32)}
    result = {
        "source": args.checkpoint,
        "source_iteration": ck["iteration"],
        "actor_called": False,
        "warm_start_actor": False,
        "stage_balanced": False,
        "simulator_transitions": 0,
        "baseline": evaluate(model, bridge, batches),
        "cases": [],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_horizon = max(h for h, _ in cases)
    for horizon, weight in cases:
        model.load_state_dict(ck["model"])
        opt = torch.optim.Adam(model.parameters(), lr=conf["lr"])
        opt.load_state_dict(copy.deepcopy(ck["model_opt"]))
        # Each case draws identical start states/windows. Only the prefix used
        # for loss differs, so long-window eligibility cannot change sampling.
        torch.manual_seed(args.seed + 1)
        begin = timestamp(args.device)
        for update in range(args.updates):
            seq = replay.sample(args.batch, train_horizon)
            loss = window_loss(model, bridge, seq, horizon, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
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
        }
        result["cases"].append(row)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "source": args.checkpoint,
                "case": row,
                "contract": c,
                "config": conf,
            },
            output / f"wm_window{horizon}_weight{weight:g}.pt",
        )
        (output / "comparison.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
