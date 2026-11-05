"""Oracle WM actor with its saved observation normalizer, for unchanged evaluation."""

from torch import nn
from .models import Standardizer, mlp


def build_actor(checkpoint, num_obs, device):
    if num_obs != 179 or checkpoint.get("version") not in (1, 2, 3):
        raise ValueError(
            "World-model actor requires the 179-D Oracle observation contract"
        )
    actor = mlp(
        179,
        29,
        checkpoint["config"]["hidden"],
        layers=checkpoint["config"].get("actor_hidden_dims"),
    )
    normalizer = Standardizer(179)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    normalizer.load_state_dict(checkpoint["obs_norm"], strict=True)
    return nn.Sequential(normalizer, actor).to(device).eval()
