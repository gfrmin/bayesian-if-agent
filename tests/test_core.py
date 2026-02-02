"""Tests for core.py — GameState, Transition, Outcome, DynamicsModel, BayesianActionSelector, BayesianIFAgent."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import (
    GameState, Transition, Outcome, DynamicsModel,
    BayesianActionSelector, BayesianIFAgent,
)


def _make_state(loc, inv=(), world_hash=""):
    return GameState(loc, frozenset(inv), world_hash)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

def test_gamestate_is_hashable():
    s1 = _make_state(5, ["keys"], "abc")
    s2 = _make_state(5, ["keys"], "abc")
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


def test_gamestate_initial():
    s = GameState.initial()
    assert s.location == 0
    assert s.inventory == frozenset()
    assert s.world_hash == ""


def test_gamestate_different_hash():
    s1 = _make_state(5, ["keys"], "abc")
    s2 = _make_state(5, ["keys"], "def")
    assert s1 != s2


# ---------------------------------------------------------------------------
# Transition / Outcome
# ---------------------------------------------------------------------------

def test_transition_is_frozen():
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")
    t = Transition(s1, "go north", s2, 1.0, "You go north.")
    assert t.state == s1
    assert t.raw_observation == "You go north."
    try:
        t.reward = 5.0  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_outcome_is_frozen():
    s = _make_state(1, world_hash="a")
    o = Outcome(next_state=s, reward=5.0)
    assert o.next_state == s
    assert o.reward == 5.0


# ---------------------------------------------------------------------------
# DynamicsModel
# ---------------------------------------------------------------------------

def test_dynamics_update_and_predict():
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")

    dm.update(s1, "go", s2, 1.0, "You go north.")
    dm.update(s1, "go", s2, 1.0, "You go north again.")

    dist = dm.predict(s1, "go")
    assert (s2, 1.0) in dist
    assert dist[(s2, 1.0)] > 0.9


def test_dynamics_history_stored():
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")

    dm.update(s1, "go", s2, 1.0, "obs1")
    dm.update(s1, "go", s2, 1.0, "obs2")

    assert len(dm.history) == 2
    assert dm.history[0].raw_observation == "obs1"
    assert dm.history[1].raw_observation == "obs2"


def test_dynamics_get_outcome_counts():
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")
    s3 = _make_state(3, world_hash="c")

    dm.update(s1, "go", s2, 1.0)
    dm.update(s1, "go", s2, 1.0)
    dm.update(s1, "go", s3, 0.0)

    counts = dm.get_outcome_counts(s1, "go")
    assert counts[(s2, 1.0)] == 2
    assert counts[(s3, 0.0)] == 1


def test_dynamics_observation_count():
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")

    assert dm.observation_count(s1, "go") == 0
    dm.update(s1, "go", s2, 0.0)
    assert dm.observation_count(s1, "go") == 1


def test_dynamics_unknown_prediction_empty():
    dm = DynamicsModel()
    s = _make_state(99, world_hash="x")
    assert dm.predict(s, "jump") == {}


def test_dynamics_expected_reward():
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")

    dm.update(s1, "go", s2, 5.0)
    dm.update(s1, "go", s2, 5.0)

    reward = dm.expected_reward(s1, "go")
    assert reward > 4.0  # Should be close to 5.0


# ---------------------------------------------------------------------------
# BayesianActionSelector
# ---------------------------------------------------------------------------

def test_selector_returns_valid_action():
    dm = DynamicsModel()
    s = _make_state(1, world_hash="a")
    sel = BayesianActionSelector(dynamics=dm, exploration_weight=0.1)
    action = sel.select_action(s, ["look", "go north"], "You are in a room.")
    assert action in ["look", "go north"]


def test_selector_prefers_known_reward():
    dm = DynamicsModel()
    s = _make_state(1, world_hash="a")
    s_good = _make_state(2, world_hash="good")
    s_bad = _make_state(3, world_hash="bad")

    for _ in range(20):
        dm.update(s, "win", s_good, 10.0)
        dm.update(s, "lose", s_bad, -5.0)

    sel = BayesianActionSelector(dynamics=dm, exploration_weight=0.01)

    # Run many trials to check bias
    win_count = sum(
        1 for _ in range(100)
        if sel.select_action(s, ["win", "lose"], "room") == "win"
    )
    assert win_count > 70  # Should strongly prefer "win"


def test_selector_with_mock_sensor():
    from sensor_model import LLMSensorModel

    class MockSensor:
        def get_relevance_scores(self, obs, actions):
            return {a: 0.9 if "good" in a else 0.1 for a in actions}

    dm = DynamicsModel()
    sm = LLMSensorModel()
    sel = BayesianActionSelector(
        dynamics=dm, exploration_weight=0.1,
        sensor=MockSensor(), sensor_model=sm,
    )

    action = sel.select_action(
        _make_state(1, world_hash="a"),
        ["good action", "bad action"],
        "test",
    )
    assert action in ["good action", "bad action"]


def test_selector_observe_outcome():
    dm = DynamicsModel()
    s1 = _make_state(1, world_hash="a")
    s2 = _make_state(2, world_hash="b")

    sel = BayesianActionSelector(dynamics=dm)
    sel.observe_outcome(s1, "go", Outcome(next_state=s2, reward=1.0))
    assert dm.observation_count(s1, "go") == 1


def test_selector_no_actions():
    dm = DynamicsModel()
    sel = BayesianActionSelector(dynamics=dm)
    action = sel.select_action(_make_state(1, world_hash="a"), [], "room")
    assert action == "look"


# ---------------------------------------------------------------------------
# BayesianIFAgent
# ---------------------------------------------------------------------------

def test_agent_observe_and_act():
    agent = BayesianIFAgent()
    s0 = _make_state(1, world_hash="h0")
    agent.observe(s0, "You wake up.", 0.0)
    action = agent.act(["look", "go north", "take keys"], "You wake up.")
    assert action in ["look", "go north", "take keys"]


def test_agent_dynamics_updated_after_two_observations():
    agent = BayesianIFAgent()
    s0 = _make_state(1, world_hash="h0")
    agent.observe(s0, "Bedroom.", 0.0)
    agent.act(["go north"], "Bedroom.")

    s1 = _make_state(2, world_hash="h1")
    agent.observe(s1, "Hallway.", 0.0)

    assert len(agent.dynamics.observed_transitions) >= 1
    assert len(agent.dynamics.history) >= 1


def test_agent_get_statistics():
    agent = BayesianIFAgent()
    s0 = _make_state(1, world_hash="h0")
    agent.observe(s0, "Bedroom.", 0.0)
    stats = agent.get_statistics()
    assert "total_steps" in stats
    assert "dynamics_history_size" in stats
    assert stats["total_steps"] == 1


def test_agent_act_with_budget():
    """Verify that passing a ComputationBudget to act() works."""
    from metareason import ComputationBudget

    agent = BayesianIFAgent()
    s0 = _make_state(1, world_hash="h0")
    agent.observe(s0, "Bedroom.", 0.0)

    budget = ComputationBudget(max_iterations=3, time_value=0.01)
    action = agent.act(["look", "go north"], "Bedroom.", budget=budget)
    assert action in ["look", "go north"]


def test_agent_with_sensor_model():
    from sensor_model import LLMSensorModel

    sm = LLMSensorModel()
    agent = BayesianIFAgent(sensor_model=sm, exploration_weight=0.2)

    s0 = _make_state(1, world_hash="h0")
    agent.observe(s0, "Bedroom.", 0.0)
    agent.act(["look"], "Bedroom.")

    s1 = _make_state(2, world_hash="h1")
    agent.observe(s1, "Hallway.", 5.0)

    stats = agent.get_statistics()
    assert "sensor_stats" in stats
