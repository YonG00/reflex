"""PPO with the shared real-data AMP motion prior."""

import torch
from rsl_rl.algorithms.ppo import PPO
from reflex.amp.context import AMPContext, AMPDiscriminator as AMPDiscriminator


class AmpPPO(AMPContext, PPO):
    """PPO + AMP style reward. Call set_amp_context() before learn()."""

    amp_ready: bool = False

    def process_env_step(self, obs, rewards, dones, extras):
        if self.amp_ready:
            feats = self._features()
            pair = torch.stack([self._prev, feats], dim=1)  # (n,2,83)
            with torch.no_grad():
                logit = self._disc(self._norm(pair)).squeeze(-1)
                prob = torch.sigmoid(logit)
                style = -torch.log((1.0 - prob).clamp(min=1e-4))
                style = style.clamp(max=5.0)
            style = torch.where(self._prev_valid, style, torch.zeros_like(style))
            r = rewards.squeeze(-1) if rewards.dim() > 1 else rewards
            # w_style ramp-in (a guard for warm-started training that the AMP
            # paper does not need): early on, the policy must improve on the
            # task gradient alone; a full style reward from the first
            # iteration stalled task learning.
            ramp = min(1.0, self._iter_count / max(1, self._style_ramp_iters))
            mixed = self._w_task * r + (self._w_style * ramp) * style
            rewards = mixed.unsqueeze(-1) if rewards.dim() > 1 else mixed
            # stash policy pairs (only valid ones)
            vidx = self._prev_valid.nonzero(as_tuple=False).squeeze(-1)
            if len(vidx):
                m = len(vidx)
                idx = (
                    self._pbuf_ptr + torch.arange(m, device=self.device)
                ) % self._pbuf.shape[0]
                self._pbuf[idx] = pair[vidx]
                self._pbuf_ptr = int((self._pbuf_ptr + m) % self._pbuf.shape[0])
                self._pbuf_n = min(self._pbuf_n + m, self._pbuf.shape[0])
            self._style_ema = 0.99 * self._style_ema + 0.01 * style.mean().item()
            self._prev = feats
            done_f = dones.squeeze(-1).bool() if dones.dim() > 1 else dones.bool()
            self._prev_valid = ~done_f
        return super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        if self.amp_ready:
            self._iter_count += 1
            disc_ready = self._pbuf_n >= self._batch
            if getattr(self, "_ddp", False):
                # Multi-GPU: rank-consistent gate. pbuf fill rate differs per
                # rank (valid-pair counts depend on local dones); _disc_update
                # runs collectives, so all ranks must take the same branch.
                t = torch.tensor([1.0 if disc_ready else 0.0], device=self.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
                disc_ready = bool(t.item() > 0.5)
            if disc_ready:
                # adaptive balance: when the disc is winning hard, let the
                # policy catch up (cheap stand-in for official interleaving)
                if self._last_disc_loss >= self._disc_skip_below:
                    self._last_disc_loss = self._disc_update()
                else:
                    self._last_disc_loss = min(1.0, self._last_disc_loss + 0.05)
                ramp = min(1.0, self._iter_count / max(1, self._style_ramp_iters))
                print(
                    f"[AMP] disc_loss {self._last_disc_loss:.3f} "
                    f"style_ema {self._style_ema:.3f} ramp {ramp:.2f} "
                    f"pbuf {self._pbuf_n}",
                    flush=True,
                )
            if getattr(self, "_ddp_probe", False) and (
                self._iter_count == 1 or self._iter_count % self._ddp_probe_every == 0
            ):
                # Multi-GPU probe: cross-rank sync check. Param sums must be
                # IDENTICAL on every rank (disc after this iter's disc update,
                # actor-critic after the previous iter's synced PPO update).
                with torch.no_grad():
                    dsum = sum(p.double().sum().item() for p in self._disc.parameters())
                    acsum = sum(
                        p.double().sum().item() for p in self.policy.parameters()
                    )
                print(
                    f"[amp_ddp_probe] rank {getattr(self, '_ddp_rank', 0)} "
                    f"iter {self._iter_count} disc_sum {dsum:.10e} "
                    f"ac_sum {acsum:.10e}",
                    flush=True,
                )
        return super().update()
