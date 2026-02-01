"""Tests for metareason.py — deliberation budget."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import GameState, DynamicsModel, BeliefState, ActionSelector
from metareason import (
    MetaState,
    ComputationBudget,
    value_of_thinking,
    should_keep_thinking,
    deliberate,
    _action_entropy,
)


def _make_state(loc, inv=(), flags=()):
    return GameState(loc, frozenset(inv), frozenset(flags))


# ---------------------------------------------------------------------------
# _action_entropy
# ---------------------------------------------------------------------------

def test_entropy_equal_values():
    """Equal action values → maximum entropy."""
    import math
    e = _action_entropy({"a": 0.0, "b": 0.0, "c": 0.0})
    # Uniform over 3 → log(3)
    assert abs(e - math.log(3)) < 0.01


def test_entropy_dominant_value():
    """One action much better → low entropy."""
    e = _action_entropy({"a": 100.0, "b": 0.0})
    assert e < 0.1


def test_entropy_empty():
    assert _action_entropy({}) == float('inf')


# ---------------------------------------------------------------------------
# value_of_thinking / should_keep_thinking
# ---------------------------------------------------------------------------

def test_value_of_thinking_decreases_with_computation():
    meta = MetaState(decision_entropy=1.0, computation_done=0)
    v0 = value_of_thinking(meta, time_value=0.001)

    meta_later = MetaState(decision_entropy=1.0, computation_done=5)
    v5 = value_of_thinking(meta_later, time_value=0.001)

    assert v0 > v5  # diminishing returns


def test_should_keep_thinking_stops_eventually():
    budget = ComputationBudget(max_iterations=100, time_value=0.01)
    meta = MetaState(decision_entropy=1.0, computation_done=0)

    # Simulate increasing computation
    stopped = False
    for i in range(100):
        meta.computation_done = i
        if not should_keep_thinking(meta, budget):
            stopped = True
            break

    assert stopped, "Should eventually stop thinking"


def test_should_keep_thinking_respects_max_iterations():
    budget = ComputationBudget(max_iterations=3, time_value=0.0)
    meta = MetaState(decision_entropy=10.0, computation_done=3)
    assert not should_keep_thinking(meta, budget)


# ---------------------------------------------------------------------------
# deliberate
# ---------------------------------------------------------------------------

def test_deliberate_returns_valid_action():
    dm = DynamicsModel()
    s = _make_state("room")
    belief = BeliefState(s)
    selector = ActionSelector()
    budget = ComputationBudget(max_iterations=5, time_value=0.01)

    action, meta = deliberate(belief, dm, selector, ["look", "go"], budget)
    assert action in ["look", "go"]
    assert meta.computation_done >= 1


def test_deliberate_easy_decision_fewer_steps():
    """When one action is clearly better, deliberation should be shorter."""
    dm = DynamicsModel()
    s = _make_state("room")
    s_good = _make_state("good")
    s_bad = _make_state("bad")

    # Make "win" clearly better
    for _ in range(20):
        dm.update(s, "win", s_good, 10.0)
        dm.update(s, "lose", s_bad, -5.0)

    belief = BeliefState(s)
    selector = ActionSelector()
    budget = ComputationBudget(max_iterations=50, time_value=0.01)

    action_easy, meta_easy = deliberate(
        belief, dm, selector, ["win", "lose"], budget
    )

    # Now test a hard decision (similar values)
    dm2 = DynamicsModel()
    s_x = _make_state("x")
    s_y = _make_state("y")
    for _ in range(20):
        dm2.update(s, "opt_a", s_x, 1.0)
        dm2.update(s, "opt_b", s_y, 1.01)

    budget2 = ComputationBudget(max_iterations=50, time_value=0.01)
    action_hard, meta_hard = deliberate(
        belief, dm2, selector, ["opt_a", "opt_b"], budget2
    )

    # Easy decision should use fewer steps (or equal in edge case)
    assert meta_easy.computation_done <= meta_hard.computation_done


def test_deliberate_budget_exhaustion():
    """With max_iterations=1, should stop after 1 iteration."""
    dm = DynamicsModel()
    s = _make_state("room")
    belief = BeliefState(s)
    selector = ActionSelector()
    budget = ComputationBudget(max_iterations=1, time_value=0.0)

    action, meta = deliberate(belief, dm, selector, ["a", "b"], budget)
    assert meta.computation_done == 1


def test_deliberate_empty_actions():
    dm = DynamicsModel()
    belief = BeliefState(_make_state("room"))
    selector = ActionSelector()
    budget = ComputationBudget()

    action, meta = deliberate(belief, dm, selector, [], budget)
    assert action == ""
    assert meta.computation_done == 0
