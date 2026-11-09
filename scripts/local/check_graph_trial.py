"""Report strict failures separately from controlled-training update checks.

Does not rewrite tolerances or promote a candidate into default configuration.
Only the caller's explicit --allow-feature-trial permits recorded feature diffs.
"""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("report", type=Path)
p.add_argument("config", type=Path)
p.add_argument("--allow-feature-trial", action="store_true")
a = p.parse_args()
r = json.loads(a.report.read_text())
c = json.loads(a.config.read_text())
assert r["completed"] and not r.get("error") and r["performance_usable"]
assert all(r["config"].get(k) == v for k, v in c.items()), (
    "Config differs from tested candidate"
)
assert r["continuous_update_numeric_passed"], (
    "Consecutive parameter/gradient/AMP updates differ"
)
assert not any(r["profile_compile_delta"].values())
failures = r["variants"]["graph_wm"]["post_wm_trajectory_and_gradient"]["failures"]
assert r["variants"]["graph_wm"]["actual_parameters_and_gradients"]["passed"]
if not r["numeric_passed"]:
    assert a.allow_feature_trial, (
        "Strict trace comparison failed; explicit trial choice required"
    )
    assert set(failures) <= {"critic_obs", "style"}, failures
reference = r["variants"]["eager_wm"]["seconds"]
candidate = r["variants"]["graph_wm"]["seconds"]
assert candidate["wm10_amp_seconds"]["median"] < reference["wm10_amp_seconds"]["median"]
assert candidate["complete_seconds"]["median"] < reference["complete_seconds"]["median"]
print(
    json.dumps(
        {
            "strict_numeric_passed": r["numeric_passed"],
            "continuous_updates_passed": True,
            "recorded_feature_failures": failures,
            "cuda_graph_launch_events": r.get("cuda_graph_launch_events"),
            "requires_live_graph_replay_confirmation": not r.get(
                "cuda_graph_launch_events", 0
            ),
            "reference_seconds": reference,
            "candidate_seconds": candidate,
            "scope": "isolated training trial, not default promotion",
        },
        indent=2,
    )
)
