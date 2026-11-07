import pytest
from reflex.world_model.joint_budget import JointBudget


def test_warmup_and_first_cold_update_are_outside_joint_budget():
    budget = JointBudget(1800)
    assert not budget.expired(10000)
    assert budget.elapsed(10000) == 0
    assert budget.actor_completed(10000)
    assert not budget.actor_completed(11000)
    assert not budget.expired(11799)
    assert budget.expired(11800)
    assert budget.elapsed(11800) == 1800


def test_optional_budget_does_not_limit_existing_runs():
    budget = JointBudget()
    budget.actor_completed(10)
    assert not budget.expired(100000)
    with pytest.raises(ValueError):
        JointBudget(0)
