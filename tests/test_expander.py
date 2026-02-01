"""Tests for expander.py — propose, evaluate, apply expansions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import GameState, Transition, DynamicsModel
from contradiction import Contradiction
from expander import (
    ExpansionProposal,
    propose_expansion,
    evaluate_expansion,
    apply_expansion,
    rebuild_dynamics,
    _diff_observations,
)


def _make_state(loc, inv=(), flags=()):
    return GameState(loc, frozenset(inv), frozenset(flags))


# ---------------------------------------------------------------------------
# _diff_observations
# ---------------------------------------------------------------------------

def test_diff_observations_finds_keyword():
    group_a = ["The door is open and you walk through"]
    group_b = ["The door is locked and you cannot pass"]
    result = _diff_observations(group_a, group_b)
    assert result is not None
    # Should find a distinguishing word (e.g., "open" or "locked")
    assert result in ("open", "locked", "walk", "through", "cannot", "pass")


def test_diff_observations_returns_none_for_identical():
    group_a = ["hello world"]
    group_b = ["hello world"]
    result = _diff_observations(group_a, group_b)
    assert result is None


# ---------------------------------------------------------------------------
# propose_expansion
# ---------------------------------------------------------------------------

def test_propose_expansion_returns_proposal():
    s = _make_state("room")
    s_a = _make_state("north")
    s_b = _make_state("south")

    history = [
        Transition(s, "go", s_a, 0.0, "You go north through the open door."),
        Transition(s, "go", s_b, 0.0, "The door is locked. You go south."),
    ]

    contradiction = Contradiction(
        state=s, action="go",
        outcome_a=(s_a, 0.0), outcome_b=(s_b, 0.0),
        count_a=1, count_b=1,
    )

    proposal = propose_expansion(contradiction, history)
    assert proposal is not None
    assert isinstance(proposal, ExpansionProposal)
    assert proposal.estimated_benefit > 0


def test_propose_expansion_returns_none_no_observations():
    s = _make_state("room")
    s_a = _make_state("a")
    s_b = _make_state("b")

    # History has no matching transitions with raw observations
    history = [
        Transition(s, "go", s_a, 0.0, ""),
        Transition(s, "go", s_b, 0.0, ""),
    ]

    contradiction = Contradiction(
        state=s, action="go",
        outcome_a=(s_a, 0.0), outcome_b=(s_b, 0.0),
        count_a=1, count_b=1,
    )

    assert propose_expansion(contradiction, history) is None


# ---------------------------------------------------------------------------
# evaluate_expansion
# ---------------------------------------------------------------------------

def test_evaluate_expansion_accepts_when_beneficial():
    proposal = ExpansionProposal(
        flag_name="test_flag",
        source_contradiction=Contradiction(
            _make_state("r"), "a",
            (_make_state("x"), 0.0), (_make_state("y"), 0.0),
            3, 2
        ),
        estimated_benefit=5.0,
        estimated_cost=1.0,
    )
    # Many remaining episodes → benefit outweighs cost
    assert evaluate_expansion(proposal, remaining_episodes=20) is True


def test_evaluate_expansion_rejects_when_costly():
    proposal = ExpansionProposal(
        flag_name="test_flag",
        source_contradiction=Contradiction(
            _make_state("r"), "a",
            (_make_state("x"), 0.0), (_make_state("y"), 0.0),
            1, 1
        ),
        estimated_benefit=2.0,
        estimated_cost=1.0,
    )
    # Very few remaining episodes → not worth it
    assert evaluate_expansion(proposal, remaining_episodes=1, complexity_cost=10.0) is False


# ---------------------------------------------------------------------------
# apply_expansion + rebuild_dynamics
# ---------------------------------------------------------------------------

def test_apply_expansion_reparses_history():
    s1 = _make_state("a")
    s2 = _make_state("b")

    history = [
        Transition(s1, "go", s2, 1.0, "You go to b and see a key."),
    ]

    def reparse(obs, prev):
        flags = frozenset(["has_key"]) if "key" in obs else frozenset()
        return GameState("b", frozenset(), flags)

    new_history = apply_expansion("has_key", history, reparse)
    assert len(new_history) == 1
    assert "has_key" in new_history[0].next_state.flags


def test_rebuild_dynamics_from_transitions():
    s1 = _make_state("a")
    s2 = _make_state("b")

    transitions = [
        Transition(s1, "go", s2, 1.0, "obs1"),
        Transition(s1, "go", s2, 1.0, "obs2"),
    ]

    dm = rebuild_dynamics(transitions, pseudocount=0.1)
    assert len(dm.history) == 2
    assert dm.observation_count(s1, "go") == 2

    dist = dm.predict(s1, "go")
    assert (s2, 1.0) in dist
    assert dist[(s2, 1.0)] > 0.9
