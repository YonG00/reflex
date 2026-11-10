"""New escape-only mode. The old impact term is never registered here."""
import os
from isaaclab.managers import RewardTermCfg
from isaaclab.sensors import ContactSensorCfg
from reflex.envs.escape_penalty_core import EscapeConfig
from reflex.envs.escape_penalty import escape_penalty
from reflex.envs.escape_penalty_sensor import EscapeRobotContactSensor


def configure_escape_penalty(cfg):
    if hasattr(cfg.rewards, 'impact_buffer'):
        cfg.rewards.impact_buffer = None
    if hasattr(cfg.scene, 'impact_box_arm'):
        cfg.scene.impact_box_arm = None
    params = {name: float(os.environ.get('BOX_ESCAPE_'+name.upper(), default))
              for name, default in vars(EscapeConfig()).items()}
    EscapeConfig(**params)
    params.update(log_every=int(os.environ.get('BOX_ESCAPE_LOG_EVERY','50')),
                  log_path=os.environ.get('BOX_ESCAPE_LOG_PATH',f'outputs/escape_penalty/escape_{os.getpid()}.jsonl'))
    def reporting(spawn):
        spawn.activate_contact_sensors = True
        for child in getattr(spawn, 'assets_cfg', ()):
            reporting(child)
    reporting(cfg.scene.balloon.spawn)
    cfg.scene.escape_box_robot = ContactSensorCfg(class_type=EscapeRobotContactSensor,
        prim_path=cfg.scene.balloon.prim_path,
        filter_prim_paths_expr=[cfg.scene.robot.prim_path+'/.*'],
        history_length=cfg.decimation, update_period=0., debug_vis=False)
    cfg.rewards.escape_penalty = RewardTermCfg(func=escape_penalty, weight=1., params=params)
    print('[escape_penalty] legacy impact penalties disabled; one-shot failure '+str(params),flush=True)
