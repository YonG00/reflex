"""CPU contract checks with Isaac Lab interfaces stubbed; no simulator required."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
ENVS = ROOT / "source/reflex/reflex/envs"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conjugate(q):
    return q * q.new_tensor([1, -1, -1, -1])


def multiply(a, b):
    aw, av, bw, bv = a[:, :1], a[:, 1:], b[:, :1], b[:, 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def inverse(q, v):
    pure = torch.cat((torch.zeros_like(v[:, :1]), v), -1)
    return multiply(multiply(conjugate(q), pure), q)[:, 1:]


def matrix(q):
    axes = torch.eye(3, dtype=q.dtype).expand(len(q), 3, 3)
    return torch.stack([inverse(conjugate(q), axes[:, i]) for i in range(3)], -1)


def assert_close(actual, expected, atol=1e-5, rtol=1e-5):
    if actual.shape != expected.shape or not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{actual} != {expected}")


class BoxInfoTest(unittest.TestCase):
    def setUp(self):
        math = types.ModuleType("isaaclab.utils.math")
        math.quat_conjugate, math.quat_mul = conjugate, multiply
        math.quat_apply_inverse, math.matrix_from_quat = inverse, matrix
        self.stub = patch.dict(sys.modules, {"isaaclab.utils.math": math})
        self.stub.start()
        self.addCleanup(self.stub.stop)
        self.terms = load("box_info_terms", ENVS / "box_info_terms.py")
        self.masses = torch.tensor([[0.2], [1.3]])
        data = types.SimpleNamespace(
            root_quat_w=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
            root_ang_vel_w=torch.tensor([[1., 2., 3.], [4., 5., 6.]]),
            root_pos_w=torch.zeros(2, 3))
        robot = types.SimpleNamespace(data=types.SimpleNamespace(
            root_quat_w=data.root_quat_w.clone()))
        box = types.SimpleNamespace(data=data, root_physx_view=types.SimpleNamespace(
            get_masses=lambda: self.masses))
        self.env = types.SimpleNamespace(num_envs=2, scene={"robot": robot, "balloon": box},
            _box_half_extents=torch.tensor([[.1, .2, .3], [.4, .5, .6]]))

    def test_identity_and_relative_orientation(self):
        expected = torch.tensor([[1., 0., 0., 1., 0., 0.]]).expand(2, 6)
        assert_close(self.terms.box_orientation_6d(self.env), expected)
        q = torch.tensor([[2**-.5, 0., 0., 2**-.5]]).expand(2, 4)
        self.env.scene["robot"].data.root_quat_w = q
        expected = torch.tensor([[0., 1., -1., 0., 0., 0.]]).expand(2, 6)
        assert_close(self.terms.box_orientation_6d(self.env), expected, atol=1e-6, rtol=0)
        self.env.scene["balloon"].data.root_quat_w = q
        assert_close(self.terms.box_orientation_6d(self.env),
            torch.tensor([[1., 0., 0., 1., 0., 0.]]).expand(2, 6), atol=1e-6, rtol=0)

    def test_angular_velocity_frame(self):
        self.env.scene["robot"].data.root_quat_w[:] = torch.tensor([2**-.5, 0., 0., 2**-.5])
        assert_close(self.terms.box_angular_velocity_b(self.env),
            torch.tensor([[2., -1., 3.], [5., -4., 6.]]))

    def test_sizes_and_live_mass(self):
        assert_close(self.terms.box_dimensions(self.env),
            torch.tensor([[.2, .4, .6], [.8, 1., 1.2]]))
        assert_close(self.terms.box_mass(self.env), self.masses)
        self.masses = torch.tensor([[2.], [3.]])
        assert_close(self.terms.box_mass(self.env), self.masses)
        tensors = [fn(self.env) for fn in (self.terms.box_orientation_6d,
            self.terms.box_angular_velocity_b, self.terms.box_dimensions, self.terms.box_mass)]
        self.assertEqual(torch.cat(tensors, -1).shape, (2, 13))

    def test_size_probe_before_first_reset(self):
        del self.env._box_half_extents
        self.env.scene["balloon"].device = "cpu"
        calls = []
        def geometry(box, models, device):
            calls.append(True)
            return None, torch.tensor([[.1, .2, .3], [.4, .5, .6]]), None
        fake = types.SimpleNamespace(BOX_MODELS=(), infer_models_from_geometry=geometry)
        with patch.dict(sys.modules, {"reflex.envs.throw": fake}):
            first = self.terms.box_dimensions(self.env)
            assert_close(first, torch.tensor([[.2, .4, .6], [.8, 1., 1.2]]))
            assert_close(self.terms.box_dimensions(self.env), first)
        self.assertEqual(len(calls), 1)
        self.assertFalse(hasattr(self.env, "_box_half_extents"))

    def test_config_appends_to_both_groups(self):
        class Base:
            def __post_init__(self):
                self.observations = types.SimpleNamespace(
                    policy=types.SimpleNamespace(original="actor"),
                    critic=types.SimpleNamespace(original="critic"))
        stubs = {
            "isaaclab.managers": types.SimpleNamespace(ObservationTermCfg=lambda **kw: kw),
            "isaaclab.utils": types.SimpleNamespace(configclass=lambda cls: cls),
            "reflex.envs": types.SimpleNamespace(box_info_terms=self.terms),
            "reflex.envs.teacher_env_cfg": types.SimpleNamespace(G1BoxCatchStepCatchEnvCfg=Base),
        }
        with patch.dict(sys.modules, stubs):
            cfg = load("box_info_env_cfg", ENVS / "box_info_env_cfg.py").G1BoxCatchFullBoxInfoEnvCfg()
            cfg.__post_init__()
        for group in (cfg.observations.policy, cfg.observations.critic):
            self.assertEqual(list(vars(group)), ["original", "box_orientation_6d",
                "box_angular_velocity_b", "box_dimensions", "box_mass"])
        self.assertEqual(cfg.observations.policy.original, "actor")
        self.assertEqual(cfg.observations.critic.original, "critic")


if __name__ == "__main__":
    unittest.main()
