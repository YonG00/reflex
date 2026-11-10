import importlib.util
from pathlib import Path
from types import SimpleNamespace

path=Path(__file__).resolve().parents[1]/'source/reflex/reflex/eval/video_settings.py'
spec=importlib.util.spec_from_file_location('video_settings_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_disable_debug_markers_keeps_physics_and_reward_settings():
    cfg=SimpleNamespace(commands=SimpleNamespace(motion=SimpleNamespace(debug_vis=True)),
        scene=SimpleNamespace(contact=SimpleNamespace(debug_vis=True)),reward_weight=100.,num_envs=320)
    assert module.disable_debug_visualization(cfg)==2
    assert not cfg.commands.motion.debug_vis and not cfg.scene.contact.debug_vis
    assert cfg.reward_weight==100. and cfg.num_envs==320


def test_nested_containers_cycles_and_idempotence():
    node={'debug_vis':True,'size':3}
    cfg=SimpleNamespace(values=[node,SimpleNamespace(debug_vis=True)],other=(node,))
    cfg.self=cfg
    assert module.disable_debug_visualization(cfg)==2
    assert module.disable_debug_visualization(cfg)==0
    assert node=={'debug_vis':False,'size':3}
