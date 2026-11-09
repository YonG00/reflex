"""TrackerLab-style return-weighted reference reset sampling (not Replay sampling).

Reference: Renforce-Dynamics/trackerlab-supermimic, managers/supermimic.py.
The environment adapter owns reward timing and termination classification.
"""

import math

import torch


class ReturnResetSampler:
    def __init__(self, frames, num_envs, gamma, max_step_reward, device="cpu"):
        if (
            frames < 1
            or num_envs < 1
            or not 0 < gamma < 1
            or not math.isfinite(max_step_reward)
            or max_step_reward <= 0
        ):
            raise ValueError("Invalid return reset sampler specification")
        self.frames, self.gamma = frames, gamma
        self.max_return = max_step_reward / (1 - gamma)
        self.bin_start = torch.arange(0, frames, 50, device=device)
        self.returns = torch.zeros(len(self.bin_start), device=device)
        self.past = torch.full_like(self.returns, 1 / len(self.returns))
        self.reset_bin = torch.zeros(num_envs, dtype=torch.long, device=device)

    @torch.no_grad()
    def sample(self, env_ids):
        soft = torch.softmax(-25 * self.returns / self.max_return, dim=0)
        self.past.mul_(0.999).add_(soft, alpha=0.001)
        probs = (
            0.5 / len(soft)
            + 0.4 * soft
            + 0.1 * self.past / self.past.sum().clamp_min(1e-8)
        )
        bins = torch.multinomial(probs, len(env_ids), replacement=True)
        start = self.bin_start[bins]
        span = (self.frames - start).clamp(max=50)
        frames = start + (torch.rand(len(env_ids), device=start.device) * span).long()
        self.reset_bin[env_ids] = bins
        return frames

    @torch.no_grad()
    def update(self, env_ids, terminated, lengths, undiscounted, discounted):
        lengths = lengths.float()
        bootstrap = torch.where(
            terminated,
            0.0,
            self.gamma**lengths
            * undiscounted
            / lengths.clamp_min(1)
            / (1 - self.gamma),
        )
        values = discounted + bootstrap
        bins = self.reset_bin[env_ids]
        sums = torch.zeros_like(self.returns).index_add_(0, bins, values)
        counts = torch.bincount(bins, minlength=len(self.returns)).to(sums.dtype)
        observed = counts > 0
        self.returns[observed] = (
            0.8 * self.returns[observed] + 0.2 * (sums / counts.clamp_min(1))[observed]
        )


class NativeReturnReset:
    """Replace only reference-frame selection; original teleport/events still run."""

    def __init__(self, env, gamma, max_step_reward):
        self.env = env
        self.command = env.command_manager.get_term("motion")
        self.sampler = ReturnResetSampler(
            self.command.motion.time_step_total,
            env.num_envs,
            gamma,
            max_step_reward,
            env.device,
        )
        self.lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.total = torch.zeros(env.num_envs, device=env.device)
        self.discounted = torch.zeros_like(self.total)
        self.terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        # _resample_command retains its full original reset and state-write path.
        self.original_sampling = self.command._adaptive_sampling
        self.command._adaptive_sampling = self.resample
        self.original_reward = env.reward_manager.compute
        env.reward_manager.compute = self.reward

    @torch.no_grad()
    def reward(self, dt):
        value = self.original_reward(dt)
        self.discounted += self.sampler.gamma**self.lengths * value
        self.total += value
        self.lengths += 1
        self.terminated.copy_(self.env.termination_manager.terminated)
        return value

    @torch.no_grad()
    def resample(self, env_ids):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.total.device)
        if not len(ids):
            return
        observed = ids[self.lengths[ids] > 0]
        if len(observed):
            self.sampler.update(
                observed,
                self.terminated[observed],
                self.lengths[observed],
                self.total[observed],
                self.discounted[observed],
            )
        self.command.time_steps[ids] = self.sampler.sample(ids)
        self.lengths[ids] = 0
        self.total[ids] = 0
        self.discounted[ids] = 0
        self.terminated[ids] = False

    def state_dict(self):
        return {
            "schema": 1,
            "frames": self.sampler.frames,
            "gamma": self.sampler.gamma,
            "max_return": self.sampler.max_return,
            "returns": self.sampler.returns.clone(),
            "past": self.sampler.past.clone(),
        }

    def load_state_dict(self, state):
        if (
            state["schema"] != 1
            or state["frames"] != self.sampler.frames
            or state["gamma"] != self.sampler.gamma
            or state["max_return"] != self.sampler.max_return
        ):
            raise ValueError("Incompatible return reset sampler checkpoint")
        validated = {}
        for name in ("returns", "past"):
            target = getattr(self.sampler, name)
            source = state[name].to(target)
            if source.shape != target.shape or not torch.isfinite(source).all():
                raise ValueError("Invalid return reset sampler buffers")
            if name == "past" and ((source < 0).any() or source.sum() <= 0):
                raise ValueError("Invalid historical reset probabilities")
            validated[name] = source
        for name, source in validated.items():
            getattr(self.sampler, name).copy_(source)
        # Live PhysX state is not restored by WM checkpoints. Do not restore
        # partial returns/reset bins belonging to the old simulator episodes.
