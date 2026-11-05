"""Shared AMP trained on real transitions; imagined inputs keep their gradients."""

from types import SimpleNamespace
import torch
from reflex.amp.context import AMPContext


class MotionPrior(AMPContext):
    def __init__(self, dataset, device, batch=1024, capacity=200000, disc_steps=4):
        self.device = device
        self.set_amp_context(
            SimpleNamespace(num_envs=1),
            dataset,
            batch=batch,
            buf_size=capacity,
            disc_steps=disc_steps,
        )

    @torch.no_grad()
    def add_real(self, pairs):
        pairs = pairs[-len(self._pbuf) :]
        n = len(pairs)
        idx = (torch.arange(n, device=self.device) + self._pbuf_ptr) % len(self._pbuf)
        self._pbuf[idx] = pairs
        self._pbuf_ptr = (self._pbuf_ptr + n) % len(self._pbuf)
        self._pbuf_n = min(self._pbuf_n + n, len(self._pbuf))

    def train_discriminator(self):
        self._iter_count += 1
        if self._pbuf_n >= self._batch:
            if self._last_disc_loss >= self._disc_skip_below:
                self._last_disc_loss = self._disc_update()
            else:
                self._last_disc_loss = min(1.0, self._last_disc_loss + 0.05)
        return self._last_disc_loss

    def reward(self, previous, current):
        pairs = torch.stack((previous, current), -2)
        logit = self._disc(self._norm(pairs)).squeeze(-1)
        return (-torch.log((1 - torch.sigmoid(logit)).clamp_min(1e-4))).clamp(max=5)

    @property
    def ramp(self):
        return min(1.0, self._iter_count / max(1, self._style_ramp_iters))
