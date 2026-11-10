"""Three staged positive-reward ablations; no negative escape term registered."""
import os
import math
from isaaclab.managers import RewardTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from reflex.envs.escape_penalty_sensor import EscapeRobotContactSensor
from reflex.envs import buffer_control as terms


class BufferContactSensor(EscapeRobotContactSensor):
    def _filter_paths(self):
        return super()._filter_paths()+[self.cfg.ground_prim_path]


@configclass
class BufferContactSensorCfg(ContactSensorCfg):
    ground_prim_path: str = '/World/ground'


def configure_buffer_control(cfg):
    stage=int(os.environ.get('BOX_BUFFER_STAGE','0'))
    if stage not in (1,2,3):raise ValueError('BOX_BUFFER_STAGE must be 1,2,3')
    # Only retire the two newly added negative terms, never other original terms.
    for name in ('escape_penalty','impact_buffer'):
        if hasattr(cfg.rewards,name):setattr(cfg.rewards,name,None)
    def reporting(spawn):
        spawn.activate_contact_sensors=True
        for child in getattr(spawn,'assets_cfg',()):reporting(child)
    reporting(cfg.scene.balloon.spawn)
    cfg.scene.buffer_box_contacts=BufferContactSensorCfg(class_type=BufferContactSensor,
        prim_path=cfg.scene.balloon.prim_path,
        filter_prim_paths_expr=[cfg.scene.robot.prim_path+'/.*'],
        ground_prim_path=cfg.scene.terrain.prim_path,history_length=cfg.decimation,
        update_period=0.,debug_vis=False)
    cfg.rewards.torso_asym.func=terms.buffer_torso_asym
    cfg.rewards.upright_hold.func=terms.buffer_upright_hold
    intake_weight=float(os.environ.get('BOX_BUFFER_INTAKE_WEIGHT','.75'))
    retain_weight=float(os.environ.get('BOX_BUFFER_RETAIN_WEIGHT','.5'))
    if not all(math.isfinite(v) and v>=0 for v in (intake_weight,retain_weight)):
        raise ValueError('buffer weights must be finite and nonnegative')
    cfg.rewards.compliant_intake.func=terms.buffer_compliant_intake
    cfg.rewards.compliant_intake.params={}
    cfg.rewards.compliant_intake.weight=intake_weight if stage>=2 else 0.
    if stage>=3:
        cfg.rewards.retain_reward=RewardTermCfg(func=terms.retain_reward,weight=retain_weight)
    print(f'[buffer_control] stage={stage}: escape/impact absent; intake='
          f'{cfg.rewards.compliant_intake.weight}, retain={retain_weight if stage>=3 else 0}; '
          'original AMP, tracking, terminations and action scales unchanged',flush=True)
