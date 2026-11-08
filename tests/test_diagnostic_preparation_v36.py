import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from reflex.world_model.diagnostic_watch_v35 import atomic_json, cold_watch
from reflex.world_model.physical_full_v35 import check
from test_actor_diagnostics_v31 import assert_tree_equal, fixture


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch._dynamo.reset()


def test_atomic_progress_hooks_restore_even_on_error(tmp_path):
    from types import SimpleNamespace

    original = lambda x: x * 2
    t = SimpleNamespace(_fk=original, _v24_heads=None)
    path = tmp_path / "progress.json"
    with (
        pytest.raises(RuntimeError),
        cold_watch(
            t, "warm", lambda value: atomic_json(path, value), tmp_path / "stack.log"
        ),
    ):
        assert t._fk(torch.tensor(2)) == 4
        assert json.loads(path.read_text())["operation_stack"][-1] == "_fk"
        raise RuntimeError("injected")
    assert t._fk is original
    assert not hasattr(t, "_v33_phase_recorder")
    assert not list(tmp_path.glob("*.tmp"))


def test_wm_only_never_updates_actor(tmp_path, monkeypatch):
    t = fixture(tmp_path)
    t.cfg.model_window = 8
    t.cfg.compile_backend = "aot_eager"
    monkeypatch.setattr(t, "update_actor", lambda: pytest.fail("WM-only called Actor"))
    report = check(t, tmp_path / "wm", repeats=1, warmups=1, stage="wm")
    assert report["completed"] and report["candidate_wm_numeric_passed"]
    assert "candidate_numeric_passed" not in report


def test_actor_only_never_runs_wm_and_failure_preserves_progress(tmp_path, monkeypatch):
    t = fixture(tmp_path)
    t.cfg.compile_backend = "aot_eager"
    monkeypatch.setattr(
        t, "v30_model_rollout", lambda seq: pytest.fail("Actor-only called WM")
    )

    def fail():
        raise RuntimeError("warm failure")

    monkeypatch.setattr(t, "update_actor", fail)
    with pytest.raises(RuntimeError, match="warm failure"):
        check(t, tmp_path / "actor", repeats=1, warmups=1, stage="actor")
    saved = json.loads((tmp_path / "actor/physical_full_v35.json").read_text())
    assert not saved["completed"] and saved["model_restored"]
    assert saved["current_stage"] == "warm/eager_physical"


def test_saved_wm_results_survive_actor_failure(tmp_path, monkeypatch):
    t = fixture(tmp_path)
    t.cfg.model_window = 8
    t.cfg.compile_backend = "aot_eager"
    monkeypatch.setattr(
        t, "update_actor", lambda: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError, match="stop"):
        check(t, tmp_path / "both", repeats=1, warmups=1)
    saved = json.loads((tmp_path / "both/physical_full_v35.json").read_text())
    assert saved["variants"]["native_split"]["wm_h8_gradient_and_adam"]["passed"]
    assert not saved["completed"]


def test_group_timeout_stops_child_after_parent_exit(tmp_path):
    supervisor = Path(__file__).parents[1] / "scripts/local/run_diagnostic_group.py"
    beat = tmp_path / "heartbeat"
    child = f"import signal,time,pathlib;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);p=pathlib.Path({str(beat)!r})\nwhile True:p.write_text(str(time.time()));time.sleep(.02)"
    parent = f"import subprocess,sys,time,pathlib;subprocess.Popen([sys.executable,'-c',{child!r}]);p=pathlib.Path({str(beat)!r})\nwhile not p.exists():time.sleep(.01)"
    result = subprocess.run(
        [
            sys.executable,
            str(supervisor),
            "--timeout",
            "3",
            "--grace",
            ".15",
            "--",
            sys.executable,
            "-c",
            parent,
        ],
        check=False,
        timeout=5,
    )
    assert result.returncode == 0
    before = beat.read_text()
    time.sleep(0.15)
    assert beat.read_text() == before


def test_actual_actor_fusion_pair_and_restoration(tmp_path, monkeypatch):
    from reflex.world_model.actor_fusion_study_v36 import check as study
    from test_analytic_soft_termination import new_trainer

    source = fixture(tmp_path)
    t = new_trainer(
        source,
        compile_v24_step=True,
        v24_fusion_scope="packed_heads",
        compile_backend="aot_eager",
    )
    t.replay.load_state_dict(source.replay.state_dict())
    original = t._actor_core
    flags = t.cfg.fuse_actor_core, t.cfg.actor_core_include_fk
    weights = {k: v.clone() for k, v in t.actor.state_dict().items()}
    monkeypatch.setenv("V36_STACK_SECONDS", "0")
    report = study(t, tmp_path / "fusion", repeats=1, warmups=1)
    assert (
        report["completed"]
        and report["numeric_passed"]
        and report["performance_usable"]
    ), report
    assert t._actor_core is original
    assert flags == (t.cfg.fuse_actor_core, t.cfg.actor_core_include_fk)
    assert_tree_equal(t.actor.state_dict(), weights)
    assert "update_risk" in report["variants"]["fused_no_fk"]


def test_group_actual_timeout_releases_private_child(tmp_path):
    supervisor = Path(__file__).parents[1] / "scripts/local/run_diagnostic_group.py"
    heartbeat = tmp_path / "timeout_heartbeat"
    code = f"import signal,time,pathlib;signal.signal(signal.SIGINT,signal.SIG_IGN);p=pathlib.Path({str(heartbeat)!r})\nwhile True:p.write_text(str(time.time()));time.sleep(.01)"
    result = subprocess.run(
        [
            sys.executable,
            str(supervisor),
            "--timeout",
            ".4",
            "--grace",
            ".1",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        check=False,
        timeout=3,
    )
    assert result.returncode == 124
    saved = heartbeat.read_text()
    time.sleep(0.15)
    assert heartbeat.read_text() == saved


def test_numeric_failure_is_reported_without_hiding_performance(tmp_path, monkeypatch):
    from reflex.world_model import actor_execution_v32_check as paired
    from reflex.world_model.actor_fusion_study_v36 import check as study
    from test_analytic_soft_termination import new_trainer

    source = fixture(tmp_path)
    t = new_trainer(
        source,
        compile_v24_step=True,
        v24_fusion_scope="packed_heads",
        compile_backend="aot_eager",
    )
    t.replay.load_state_dict(source.replay.state_dict())
    monkeypatch.setenv("V36_STACK_SECONDS", "0")
    monkeypatch.setattr(
        paired,
        "compare",
        lambda a, b: {
            "passed": False,
            "max_abs": {},
            "failures": {"injected": "strict mismatch"},
        },
    )
    report = study(t, tmp_path / "numeric_fail", repeats=1, warmups=1)
    assert report["completed"] and not report["numeric_passed"]
    assert report["performance_usable"]
    candidate = report["variants"]["fused_no_fk"]
    assert candidate["steady"] and candidate["numerical_risk"]["finite_all"]
    assert candidate["trajectory_and_raw_gradient"]["failures"]["injected"]


def test_supervisor_external_termination_cleans_child(tmp_path):
    supervisor = Path(__file__).parents[1] / "scripts/local/run_diagnostic_group.py"
    beat = tmp_path / "cancel_heartbeat"
    code = f"import signal,time,pathlib;signal.signal(signal.SIGINT,signal.SIG_IGN);p=pathlib.Path({str(beat)!r})\nwhile True:p.write_text(str(time.time()));time.sleep(.01)"
    process = subprocess.Popen(
        [
            sys.executable,
            str(supervisor),
            "--timeout",
            "2",
            "--grace",
            ".1",
            "--",
            sys.executable,
            "-c",
            code,
        ]
    )
    deadline = time.monotonic() + 1
    while not beat.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert beat.exists()
    process.terminate()
    assert process.wait(timeout=3) == 130
    saved = beat.read_text()
    time.sleep(0.15)
    assert beat.read_text() == saved
