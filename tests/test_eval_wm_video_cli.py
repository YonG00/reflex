import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

TOOL = Path(__file__).resolve().parents[1] / "scripts/local/eval_wm_video.py"
spec = importlib.util.spec_from_file_location("eval_video_tool", TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_command_uses_exact_checkpoint_hidden_axes_and_requested_duration(tmp_path):
    checkpoint = tmp_path / "model with space.pt"
    cmd = module.command_for(
        checkpoint, tmp_path / "config.json", tmp_path / "out", 15, 0.02, 7, 1
    )
    assert cmd[cmd.index("--checkpoint") + 1] == str(checkpoint)
    assert cmd[cmd.index("--play_steps") + 1] == "751"
    assert cmd[cmd.index("--seed") + 1] == "7"
    assert "--no_debug_vis" in cmd and "--enable_actor" not in cmd


def test_flags_are_quoted_and_other_environment_keys_rejected(tmp_path):
    f = tmp_path / "flags.env"
    value = "words & (literal)"
    module.write_flags({"BOX_TEST": value}, f)
    out = subprocess.check_output(
        ["bash", "-c", 'source "$1"; printf %s "$BOX_TEST"', "bash", str(f)], text=True
    )
    assert out == value
    with pytest.raises(ValueError):
        module.write_flags({"HOME": "bad"}, f)


def test_dry_run_extracts_checkpoint_config_without_isaac(tmp_path):
    f = tmp_path / "model.pt"
    out = tmp_path / "output with space"
    config = {"horizon": 16, "reward_profile": "standing_h16_v1"}
    torch.save(
        {
            "config": config,
            "actor": {},
            "model": {},
            "task_spec": {"dt": 0.02},
            "actor_updates": 524,
            "contract": {"flags": {"BOX_TEST": "1"}},
        },
        f,
    )
    result = subprocess.run(
        [sys.executable, str(TOOL), str(f), "--dry-run", "--output", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((out / "checkpoint_config.json").read_text()) == config
    request = json.loads((out / "evaluation_request.json").read_text())
    assert request["source"]["actor_updates"] == 524 and request["num_envs"] == 1
    assert not (out / "run.log").exists()
    assert "REFLEX_WM_ENV_FILE" in (out / "command.sh").read_text()
