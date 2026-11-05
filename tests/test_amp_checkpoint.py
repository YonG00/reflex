"""CPU regression: AMP save/load must reproduce the next discriminator update."""
import copy
from pathlib import Path
import random
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from reflex.amp.amp_ppo import AmpPPO


class AmpCheckpointTest(unittest.TestCase):
    def test_roundtrip_next_update_and_validation(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "expert.pt"
            torch.save({"features": torch.randn(24, 2, 83),
                        "stats": {"mean": torch.zeros(83), "std": torch.ones(83)}}, data)

            def build():
                alg = AmpPPO.__new__(AmpPPO)
                alg.device = "cpu"
                alg.set_amp_context(SimpleNamespace(num_envs=2), str(data),
                                    batch=4, buf_size=8, disc_steps=2)
                return alg

            a = build()
            a._pbuf.copy_(torch.randn(8, 2, 83))
            a._pbuf_n, a._pbuf_ptr = 8, 3
            a._last_disc_loss = a._disc_update()
            a._iter_count, a._style_ema = 19, 0.125
            checkpoint = Path(temp) / "amp.pt"
            torch.save(a.amp_state_dict(), checkpoint)

            expected_loss = a._disc_update()
            expected_weights = copy.deepcopy(a._disc.state_dict())
            expected_optimizer = copy.deepcopy(a._disc_opt.state_dict())
            expected_random = (random.random(), np.random.random(), torch.rand(4))

            b = build()
            b._prev_valid.fill_(True)
            b.load_amp_state_dict(torch.load(checkpoint, weights_only=False))
            self.assertEqual(b._iter_count, 19)
            self.assertEqual(b._style_ema, 0.125)
            self.assertEqual((b._pbuf_n, b._pbuf_ptr), (8, 3))
            self.assertFalse(b._prev_valid.any())
            torch.testing.assert_close(b._pbuf, a._pbuf, rtol=0, atol=0)
            self.assertEqual(b._disc_update(), expected_loss)
            for key, value in b._disc.state_dict().items():
                torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
            actual_optimizer = b._disc_opt.state_dict()
            self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
            for key, values in actual_optimizer["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, expected_optimizer["state"][key][name], rtol=0, atol=0)
            actual_random = (random.random(), np.random.random(), torch.rand(4))
            self.assertEqual(actual_random[:2], expected_random[:2])
            torch.testing.assert_close(actual_random[2], expected_random[2], rtol=0, atol=0)

            state = torch.load(checkpoint, weights_only=False)
            state["dataset_sha256"] = "wrong-dataset"
            with self.assertRaisesRegex(ValueError, "dataset checksum"):
                build().load_amp_state_dict(state)


if __name__ == "__main__":
    unittest.main()
