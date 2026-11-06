from types import SimpleNamespace as NS
import math

import torch

from reflex.world_model.standing_profile import standing_balance


def test_standing_reward_all_phases_height_and_tilt_gradient():
    height = torch.tensor([1.0, 0.8, 1.0], requires_grad=True)
    angle = torch.tensor([0.0, 0.0, math.pi / 3])
    q = torch.stack(
        [torch.cos(angle / 2), torch.sin(angle / 2), torch.zeros(3), torch.zeros(3)], -1
    )[:, None]
    p = torch.stack([torch.zeros(3), torch.zeros(3), height], -1)
    cmd = NS(
        robot_anchor_pos_w=p, anchor_pos_w=torch.tensor([[0.0, 0.0, 1.0]]).expand(3, -1)
    )
    env = NS(
        scene={"robot": NS(body_names=["torso_link"], data=NS(body_quat_w=q))},
        command_manager=NS(get_term=lambda name: cmd),
    )
    r = standing_balance(env)
    assert r[0] == 1
    assert 0 < r[1] < r[0] and r[2] < r[0]
    r.sum().backward()
    assert height.grad[1] > 0 and torch.isfinite(height.grad).all()
