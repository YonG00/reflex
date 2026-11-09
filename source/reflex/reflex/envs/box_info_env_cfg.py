"""Stage 1 observation-only ablation; inherits the official teacher unchanged."""
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from reflex.envs import box_info_terms
from reflex.envs.teacher_env_cfg import G1BoxCatchStepCatchEnvCfg


@configclass
class G1BoxCatchFullBoxInfoEnvCfg(G1BoxCatchStepCatchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Append, preserving every original input column (including actor trust).
        # The runner infers actor/critic input widths from the observation manager.
        for group in (self.observations.policy, self.observations.critic):
            group.box_orientation_6d = ObsTerm(func=box_info_terms.box_orientation_6d)
            group.box_angular_velocity_b = ObsTerm(func=box_info_terms.box_angular_velocity_b)
            group.box_dimensions = ObsTerm(func=box_info_terms.box_dimensions)
            group.box_mass = ObsTerm(func=box_info_terms.box_mass)
        print("[box-info] actor and critic: append orientation(6), angular velocity(3), "
              "dimensions(3), mass(1); original position/velocity unchanged", flush=True)
