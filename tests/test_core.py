"""Tests for core.py — GameState, Transition, DynamicsModel, BeliefState, ActionSelector, BayesianIFAgent."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import (
    GameState, Transition, DynamicsModel, BeliefState,
    ActionSelector, StateParser, BayesianIFAgent,
)


# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------

def test_gamestate_is_hashable():
    s1 = GameState("bedroom", frozenset(["keys"]), frozenset(["door_open"]))
    s2 = GameState("bedroom", frozenset(["keys"]), frozenset(["door_open"]))
    assert s1 == s2
    assert hash(s1) == hash(s2)
    assert len({s1, s2}) == 1


def test_gamestate_initial():
    s = GameState.initial()
    assert s.location == "unknown"
    assert s.inventory == frozenset()
    assert s.flags == frozenset()


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------

def test_transition_is_frozen():
    s1 = GameState("a", frozenset(), frozenset())
    s2 = GameState("b", frozenset(), frozenset())
    t = Transition(s1, "go north", s2, 1.0, "You go north.")
    assert t.state == s1
    assert t.raw_observation == "You go north."
    try:
        t.reward = 5.0  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# DynamicsModel
# ---------------------------------------------------------------------------

def test_dynamics_update_and_predict():
    dm = DynamicsModel()
    s1 = GameState("a", frozenset(), frozenset())
    s2 = GameState("b", frozenset(), frozenset())

    dm.update(s1, "go", s2, 1.0, "You go north.")
    dm.update(s1, "go", s2, 1.0, "You go north again.")

    dist = dm.predict(s1, "go")
    assert (s2, 1.0) in dist
    assert dist[(s2, 1.0)] > 0.9  # dominant outcome


def test_dynamics_history_stored():
    dm = DynamicsModel()
    s1 = GameState("a", frozenset(), frozenset())
    s2 = GameState("b", frozenset(), frozenset())

    dm.update(s1, "go", s2, 1.0, "obs1")
    dm.update(s1, "go", s2, 1.0, "obs2")

    assert len(dm.history) == 2
    assert dm.history[0].raw_observation == "obs1"
    assert dm.history[1].raw_observation == "obs2"


def test_dynamics_get_outcome_counts():
    dm = DynamicsModel()
    s1 = GameState("a", frozenset(), frozenset())
    s2 = GameState("b", frozenset(), frozenset())
    s3 = GameState("c", frozenset(), frozenset())

    dm.update(s1, "go", s2, 1.0)
    dm.update(s1, "go", s2, 1.0)
    dm.update(s1, "go", s3, 0.0)

    counts = dm.get_outcome_counts(s1, "go")
    assert counts[(s2, 1.0)] == 2
    assert counts[(s3, 0.0)] == 1


def test_dynamics_observation_count():
    dm = DynamicsModel()
    s1 = GameState("a", frozenset(), frozenset())
    s2 = GameState("b", frozenset(), frozenset())

    assert dm.observation_count(s1, "go") == 0
    dm.update(s1, "go", s2, 0.0)
    assert dm.observation_count(s1, "go") == 1


def test_dynamics_unknown_prediction_empty():
    dm = DynamicsModel()
    s = GameState("x", frozenset(), frozenset())
    assert dm.predict(s, "jump") == {}


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

def test_belief_set_and_most_likely():
    s = GameState("a", frozenset(), frozenset())
    b = BeliefState(s)
    assert b.most_likely() == s


def test_belief_update_from_observation():
    s1 = GameState("a", frozenset(), frozenset())
    s2 = GameState("b", frozenset(), frozenset())
    b = BeliefState(s1)
    b.update_from_observation(s2, confidence=0.9)
    assert b.most_likely() == s2


def test_belief_entropy_single_state():
    s = GameState("a", frozenset(), frozenset())
    b = BeliefState(s)
    assert b.entropy() < 0.01  # near zero


# ---------------------------------------------------------------------------
# ActionSelector
# ---------------------------------------------------------------------------

def test_action_selector_prefers_known_reward():
    dm = DynamicsModel()
    s = GameState("a", frozenset(), frozenset())
    s_good = GameState("good", frozenset(), frozenset())
    s_bad = GameState("bad", frozenset(), frozenset())

    for _ in range(10):
        dm.update(s, "win", s_good, 10.0)
        dm.update(s, "lose", s_bad, -5.0)

    belief = BeliefState(s)
    sel = ActionSelector(exploration_bonus=0.01)
    action, values = sel.select_action(belief, dm, ["win", "lose"])
    assert action == "win"
    assert values["win"] > values["lose"]


def test_thompson_returns_valid_action():
    dm = DynamicsModel()
    s = GameState("a", frozenset(), frozenset())
    belief = BeliefState(s)
    sel = ActionSelector()
    action = sel.thompson_sample(belief, dm, ["x", "y", "z"])
    assert action in ["x", "y", "z"]


# ---------------------------------------------------------------------------
# StateParser
# ---------------------------------------------------------------------------

def test_parser_extracts_location():
    parser = StateParser()
    state = parser.parse("You are in the bedroom. It is dark.")
    assert state.location == "bedroom"


def test_parser_extracts_flags():
    parser = StateParser()
    prev = GameState("hall", frozenset(), frozenset())
    state = parser.parse("The door is now open.", prev)
    assert "door_open" in state.flags


# ---------------------------------------------------------------------------
# BayesianIFAgent
# ---------------------------------------------------------------------------

def test_agent_observe_stores_full_observation():
    agent = BayesianIFAgent()
    long_obs = "A" * 500
    agent.observe(long_obs, 0.0)
    assert agent.history[-1]['observation'] == long_obs


def test_agent_act_returns_valid_action():
    agent = BayesianIFAgent()
    agent.observe("You are in the bedroom.", 0.0)
    action = agent.act(["look", "go north", "take keys"])
    assert action in ["look", "go north", "take keys"]


def test_agent_dynamics_updated_after_two_observations():
    agent = BayesianIFAgent()
    agent.observe("You are in the bedroom.", 0.0)
    agent.act(["go north"])
    agent.observe("You are in the hallway.", 0.0)

    assert len(agent.dynamics.observed_transitions) >= 1
    assert len(agent.dynamics.history) >= 1


def test_agent_get_statistics():
    agent = BayesianIFAgent()
    agent.observe("Bedroom.", 0.0)
    stats = agent.get_statistics()
    assert 'total_steps' in stats
    assert 'dynamics_history_size' in stats
    assert stats['total_steps'] == 1


def test_agent_act_with_budget():
    """Verify that passing a ComputationBudget to act() works."""
    from metareason import ComputationBudget

    agent = BayesianIFAgent()
    agent.observe("You are in the bedroom.", 0.0)

    budget = ComputationBudget(max_iterations=3, time_value=0.01)
    action = agent.act(["look", "go north"], budget=budget)
    assert action in ["look", "go north"]
