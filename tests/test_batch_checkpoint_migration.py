import subprocess
import sys
from pathlib import Path

import torch


def test_explicit_batch_reduction_preserves_all_tensor_states(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/local/migrate_joint_batch.py"
    )
    source, target = tmp_path / "source.pt", tmp_path / "target.pt"
    state = {
        "config": {"batch": 3840, "actor_batch": 3840, "actor_microbatch": 3840},
        "learning_contract": {"state": 84, "world_envs": 4096},
        "actor_updates": 16,
    }
    for name in ("model", "actor", "critic", "model_opt", "actor_opt", "critic_opt"):
        state[name] = {"test": torch.arange(7, dtype=torch.float32)}
    torch.save(state, source)
    subprocess.run(
        [sys.executable, str(script), str(source), str(target), "--batch", "3584"],
        check=True,
        capture_output=True,
    )
    actual = torch.load(target, weights_only=False)
    for name in ("model", "actor", "critic", "model_opt", "actor_opt", "critic_opt"):
        torch.testing.assert_close(
            state[name]["test"], actual[name]["test"], rtol=0, atol=0
        )
    assert actual["learning_contract"] == state["learning_contract"]
    assert actual["actor_updates"] == 16
    assert actual["config"] == {
        "batch": 3584,
        "actor_batch": 3584,
        "actor_microbatch": 3584,
    }
    assert target.with_suffix(".migration.json").is_file()
    # Never overwrite an existing recovered model.
    result = subprocess.run(
        [sys.executable, str(script), str(source), str(target), "--batch", "3584"],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
