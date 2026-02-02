"""Tests for contradiction.py — detect_contradictions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import GameState, DynamicsModel
from contradiction import detect_contradictions, Contradiction


def _make_state(loc, inv=(), world_hash=""):
    return GameState(loc, frozenset(inv), world_hash)


# ---------------------------------------------------------------------------
# Deterministic dynamics -> no contradictions
# ---------------------------------------------------------------------------

def test_no_contradictions_deterministic():
    """Same (state, action) always produces the same outcome."""
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")

    for _ in range(5):
        dm.update(s1, "go", s2, 1.0, "obs")

    assert detect_contradictions(dm) == []


# ---------------------------------------------------------------------------
# Coarse state -> contradictions detected
# ---------------------------------------------------------------------------

def test_contradictions_detected_different_outcomes():
    """Same (state, action) with different next_states triggers contradiction."""
    dm = DynamicsModel()
    s = _make_state(1, world_hash="room")
    s_a = _make_state(2, world_hash="room_a")
    s_b = _make_state(3, world_hash="room_b")

    dm.update(s, "go", s_a, 0.0, "You enter room A.")
    dm.update(s, "go", s_b, 0.0, "You enter room B.")

    contradictions = detect_contradictions(dm, threshold=1)
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.state == s
    assert c.action == "go"
    assert {c.outcome_a, c.outcome_b} == {(s_a, 0.0), (s_b, 0.0)}


def test_contradictions_threshold():
    """Outcomes below threshold count are ignored."""
    dm = DynamicsModel()
    s = _make_state(1, world_hash="room")
    s_a = _make_state(2, world_hash="a")
    s_b = _make_state(3, world_hash="b")

    dm.update(s, "go", s_a, 0.0, "a")
    dm.update(s, "go", s_a, 0.0, "a")
    dm.update(s, "go", s_b, 0.0, "b")  # only 1 time

    # threshold=2: s_b doesn't meet it
    assert detect_contradictions(dm, threshold=2) == []

    # threshold=1: both meet it
    assert len(detect_contradictions(dm, threshold=1)) == 1


# ---------------------------------------------------------------------------
# Multiple (state, action) pairs
# ---------------------------------------------------------------------------

def test_multiple_contradictions():
    """Multiple (state, action) pairs can each have contradictions."""
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="r1")
    s2 = _make_state(2, world_hash="r2")
    out_a = _make_state(10, world_hash="a")
    out_b = _make_state(11, world_hash="b")
    out_c = _make_state(12, world_hash="c")
    out_d = _make_state(13, world_hash="d")

    dm.update(s1, "go", out_a, 0.0, "a")
    dm.update(s1, "go", out_b, 0.0, "b")
    dm.update(s2, "look", out_c, 0.0, "c")
    dm.update(s2, "look", out_d, 0.0, "d")

    contradictions = detect_contradictions(dm)
    assert len(contradictions) == 2


def test_same_reward_different_state_is_contradiction():
    """Different next_state with same reward still counts."""
    dm = DynamicsModel()
    s = _make_state(1, world_hash="start")
    a = _make_state(2, world_hash="end_a")
    b = _make_state(3, world_hash="end_b")

    dm.update(s, "act", a, 5.0, "x")
    dm.update(s, "act", b, 5.0, "y")

    contradictions = detect_contradictions(dm)
    assert len(contradictions) == 1


def test_different_reward_same_state_is_contradiction():
    """Same next_state with different reward still counts."""
    dm = DynamicsModel()
    s = _make_state(1, world_hash="start")
    end = _make_state(2, world_hash="end")

    dm.update(s, "act", end, 0.0, "no reward")
    dm.update(s, "act", end, 5.0, "reward!")

    contradictions = detect_contradictions(dm)
    assert len(contradictions) == 1


def test_no_contradictions_different_actions():
    """Different actions from same state don't contradict each other."""
    dm = DynamicsModel()
    s = _make_state(1, world_hash="room")
    s_a = _make_state(2, world_hash="a")
    s_b = _make_state(3, world_hash="b")

    dm.update(s, "go north", s_a, 0.0, "a")
    dm.update(s, "go south", s_b, 0.0, "b")

    assert detect_contradictions(dm) == []
