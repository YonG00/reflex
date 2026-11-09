"""Opt-in sensor and single reward registration; no environment event changes."""
import os
from isaaclab.managers import RewardTermCfg
from isaaclab.sensors import ContactSensorCfg
from reflex.envs.impact_buffer import ARM_BODIES, impact_buffer_penalty
from reflex.envs.impact_buffer_core import ImpactBufferConfig
from reflex.envs.impact_buffer_sensor import ImpactPairContactSensor


def configure_impact_buffer(cfg):
    params = {name: float(os.environ.get('BOX_IMPACT_' + name.upper(), default))
              for name, default in vars(ImpactBufferConfig()).items()}
    ImpactBufferConfig(**params)  # reject invalid configuration before spawning
    params['log_every'] = int(os.environ.get('BOX_IMPACT_LOG_EVERY', '50'))
    params['log_path'] = os.environ.get('BOX_IMPACT_LOG_PATH', f'outputs/impact_buffer/impact_{os.getpid()}.jsonl')
    # Contact reporting only; collision geometry, materials and solver stay intact.
    def enable_reporting(spawn):
        spawn.activate_contact_sensors = True
        for child in getattr(spawn, 'assets_cfg', ()):
            enable_reporting(child)
    enable_reporting(cfg.scene.balloon.spawn)
    cfg.scene.impact_box_arm = ContactSensorCfg(
        class_type=ImpactPairContactSensor,
        prim_path=cfg.scene.balloon.prim_path,
        filter_prim_paths_expr=[cfg.scene.robot.prim_path + '/' + body for body in ARM_BODIES],
        history_length=cfg.decimation, update_period=0.0, debug_vis=False)
    cfg.rewards.impact_buffer = RewardTermCfg(func=impact_buffer_penalty, weight=1.0, params=params)
    print('[impact_buffer] enabled: box->8 forearm/wrist filters; reward weight=+1, '
          'negative internal components; ' + str(params), flush=True)
