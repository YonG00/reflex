"""Optional native TrackerLab networks/WM adapter, no Isaac simulation or FK.

Imports the user's existing checkout only on demand. These are explicitly
synthetic diagnostic inputs/weights, not a reproduction training experiment.
"""

import hashlib
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def factory_from_repo(repo, device, batch, seed=31001):
    repo = Path(repo).resolve()
    if not (repo / "source/trackerLab").is_dir():
        raise FileNotFoundError(repo / "source/trackerLab")
    for name in ("trackerLab", "trackerTask"):
        sys.path.insert(0, str(repo / "source" / name))
    os.environ.setdefault("SUPERMIMIC_REPO_ROOT", str(repo))
    networks = importlib.import_module("trackerLab.algorithms.supermimic.networks")
    mdp = importlib.import_module("trackerLab.tracker_env.mdp.supermimic")
    rotations = importlib.import_module("trackerLab.kinematics.supermimic.rotations")
    robots = importlib.import_module("trackerTask.supermimic.robots")
    if not Path(networks.__file__).resolve().is_relative_to(repo):
        raise ValueError(
            "Imported TrackerLab from another checkout; use a fresh process"
        )
    networks.AMP_DTYPE = None
    spec = robots.get_robot("g1")

    class TrackerStep(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(seed)
            self.model = networks.WorldModel(num_joints=spec.num_joints).to(device)
            self.actor = networks.Policy(
                99, spec.goal_frame_dim * 10, hist=10, num_joints=spec.num_joints
            ).to(device)
            # Zero-output WM initialization has a dead state->action derivative.
            # Perturb only identically-zero Linear heads, deterministically.
            self.perturbed_heads = 0
            with torch.no_grad():
                for module in self.model.modules():
                    if isinstance(module, nn.Linear) and not bool(module.weight.any()):
                        module.weight.normal_(std=1e-3)
                        self.perturbed_heads += 1
            self.model.requires_grad_(False)
            p = torch.randn(batch, 3, device=device) * 0.05
            p[:, 2] += 0.78
            R = rotations.so3_exp(torch.randn(batch, 3, device=device) * 0.1)
            v, w = (torch.randn(batch, 3, device=device) * 0.1 for _ in range(2))
            q = torch.tensor(spec.default_q, device=device).expand(batch, -1).clone()
            dq = torch.randn(batch, spec.num_joints, device=device) * 0.1
            state = (p, R, v, w, q, dq)
            xi = torch.zeros(batch, spec.context_dim, device=device)
            xi[:, 0] = 0.75
            self.data = {
                "state": state,
                "previous": torch.zeros(batch, spec.num_joints, device=device),
                "history": torch.randn(batch, 10, 99, device=device) * 0.05,
                "goal": torch.randn(batch, spec.goal_frame_dim * 10, device=device)
                * 0.05,
                "xi": mdp.normalize_xi(xi),
            }
            self.base = self.state_vector(state)
            self.wm_call = None

        def initial(self):
            return self.data["state"], self.data["previous"], self.data["history"]

        def state_vector(self, state):
            return torch.cat([v.flatten(1) for v in state], -1)

        def forward(self, state, previous, history):
            # Same fixed projection used by both drivers; no motion/FK/reward.
            obs = self.data["history"][:, -1] + 0.01 * F.pad(
                self.state_vector(state) - self.base, (0, 99 - self.base.shape[1])
            )
            obs = obs + 0.01 * F.pad(
                previous - self.data["previous"], (99 - spec.num_joints, 0)
            )
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
            raw = self.actor(history, self.data["goal"])
            wm = self.model.step if self.wm_call is None else self.wm_call
            return wm(state, raw, self.data["xi"]), raw, history

    source = Path(inspect.getsourcefile(networks.WorldModel))
    revision_error = None
    try:
        revision = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        revision = None
        revision_error = exc.stderr.strip()
    return TrackerStep, {
        "kind": "tracker_native_minimal",
        "repo": str(repo),
        "revision": revision,
        "revision_unavailable_reason": revision_error,
        "network_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": seed,
        "note": "Native Policy/WorldModel.step, synthetic inputs and diagnostic objective, no simulation/FK/reward/events; zero heads explicitly perturbed to test physical BPTT",
    }
