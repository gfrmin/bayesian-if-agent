"""Tests for core.py v3 — GameState, Transition, Outcome, DynamicsModel,
AgentBeliefs, SituationUnderstanding, OracleReliability, BeliefUpdater,
InformedActionSelector, BayesianIFAgent."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import (
    GameState, Transition, Outcome, DynamicsModel,
    AgentBeliefs, SituationUnderstanding, OracleReliability,
    BeliefUpdater, InformedActionSelector, BayesianIFAgent,
)


def _make_state(loc, inv=(), world_hash=""):
    return GameState(loc, frozenset(inv), world_hash)


# ---------------------------------------------------------------------------
# GameState (unchanged)
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
# Transition / Outcome (unchanged)
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
# DynamicsModel (unchanged)
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
# AgentBeliefs (v3)
# ---------------------------------------------------------------------------

def test_agent_beliefs_defaults():
    b = AgentBeliefs()
    assert b.location == "unknown"
    assert b.inventory == []
    assert b.overall_goal == "unknown"
    assert b.blocking_condition is None


def test_agent_beliefs_to_prompt_context():
    b = AgentBeliefs(location="bedroom", inventory=["keys", "phone"])
    ctx = b.to_prompt_context()
    assert "bedroom" in ctx
    assert "keys" in ctx
    assert "phone" in ctx


def test_agent_beliefs_to_state_key():
    b = AgentBeliefs(inventory=["keys", "wallet"])
    b._location_id = 5
    b._world_hash = "abc"
    key = b.to_state_key()
    assert isinstance(key, GameState)
    assert key.location == 5
    assert key.inventory == frozenset(["keys", "wallet"])
    assert key.world_hash == "abc"
    # Hashable
    assert hash(key) == hash(key)


def test_agent_beliefs_blocker_in_context():
    b = AgentBeliefs(blocking_condition="door is locked")
    ctx = b.to_prompt_context()
    assert "door is locked" in ctx


def test_agent_beliefs_failed_actions_in_context():
    b = AgentBeliefs()
    b.failed_actions["go east"] = "need to stand up first"
    ctx = b.to_prompt_context()
    assert "go east" in ctx
    assert "stand up" in ctx


# ---------------------------------------------------------------------------
# SituationUnderstanding (v3)
# ---------------------------------------------------------------------------

def test_situation_understanding_from_json():
    su = SituationUnderstanding.from_json({
        "overall_goal": "escape the house",
        "immediate_goal": "get dressed",
        "recommended_action": "stand up",
        "confidence": 0.8,
        "accomplished": ["woke up"],
        "alternative_actions": ["look", "take phone"],
    })
    assert su.overall_goal == "escape the house"
    assert su.immediate_goal == "get dressed"
    assert su.recommended_action == "stand up"
    assert su.confidence == 0.8
    assert "woke up" in su.accomplished
    assert "look" in su.alternative_actions


def test_situation_understanding_from_empty_json():
    su = SituationUnderstanding.from_json({})
    assert su.overall_goal is None
    assert su.recommended_action is None
    assert su.confidence == 0.5
    assert su.accomplished == []


def test_situation_understanding_defaults():
    su = SituationUnderstanding()
    assert su.overall_goal is None
    assert su.confidence == 0.5
    assert su.alternative_actions == []


# ---------------------------------------------------------------------------
# OracleReliability (v3)
# ---------------------------------------------------------------------------

def test_oracle_reliability_prior():
    r = OracleReliability()
    assert r.recommendation_accuracy == 0.5  # Prior when no data
    assert r.goal_accuracy == 0.5
    assert r.blocker_accuracy == 0.5


def test_oracle_reliability_learning():
    r = OracleReliability()
    for _ in range(7):
        r.update_recommendation(helped=True)
    for _ in range(3):
        r.update_recommendation(helped=False)
    assert 0.65 < r.recommendation_accuracy < 0.75
    assert r.recommendations_followed == 10


def test_oracle_reliability_get_summary():
    r = OracleReliability()
    r.update_recommendation(helped=True)
    summary = r.get_summary()
    assert "recommendation_accuracy" in summary
    assert "goal_accuracy" in summary
    assert "blocker_accuracy" in summary
    assert "total_recommendations" in summary
    assert summary["total_recommendations"] == 1


# ---------------------------------------------------------------------------
# BeliefUpdater (v3)
# ---------------------------------------------------------------------------

def test_belief_updater_from_understanding():
    r = OracleReliability()
    updater = BeliefUpdater(r)
    b = AgentBeliefs()

    su = SituationUnderstanding(
        overall_goal="win the game",
        immediate_goal="find the key",
        blocking_condition="door is locked",
        confidence=0.9,
        accomplished=["found map"],
    )
    b = updater.update_from_understanding(b, su)
    assert b.overall_goal == "win the game"
    assert b.current_subgoal == "find the key"
    assert b.blocking_condition == "door is locked"
    assert "found map" in b.accomplished


def test_belief_updater_low_confidence_skips_accomplishments():
    r = OracleReliability()
    updater = BeliefUpdater(r)
    b = AgentBeliefs()

    su = SituationUnderstanding(
        confidence=0.3,
        accomplished=["maybe did something"],
    )
    b = updater.update_from_understanding(b, su)
    assert "maybe did something" not in b.accomplished


def test_belief_updater_goal_not_overwritten():
    r = OracleReliability()
    updater = BeliefUpdater(r)
    b = AgentBeliefs(overall_goal="already known")

    su = SituationUnderstanding(overall_goal="new goal")
    b = updater.update_from_understanding(b, su)
    assert b.overall_goal == "already known"  # Not overwritten


def test_belief_updater_from_failure():
    r = OracleReliability()
    updater = BeliefUpdater(r)
    b = AgentBeliefs()

    analysis = {
        "failure_reason": "door is locked",
        "prerequisite": "find the key",
    }
    b = updater.update_from_failure_analysis(b, "open door", analysis)
    assert b.failed_actions["open door"] == "door is locked"
    assert b.blocking_condition == "find the key"


def test_belief_updater_from_progress():
    r = OracleReliability()
    updater = BeliefUpdater(r)
    b = AgentBeliefs(blocking_condition="some blocker")

    progress = {"made_progress": True, "accomplishment": "opened door"}
    b = updater.update_from_progress(b, progress)
    assert b.blocking_condition is None
    assert "opened door" in b.accomplished


def test_belief_updater_no_progress():
    r = OracleReliability()
    updater = BeliefUpdater(r)
    b = AgentBeliefs(blocking_condition="some blocker")

    progress = {"made_progress": False}
    b = updater.update_from_progress(b, progress)
    assert b.blocking_condition == "some blocker"  # Not cleared


# ---------------------------------------------------------------------------
# InformedActionSelector (v3)
# ---------------------------------------------------------------------------

def test_selector_returns_valid_action():
    sel = InformedActionSelector(exploration_rate=0.5)
    b = AgentBeliefs()
    action, reason = sel.select_action("You are in a room.", b, ["look", "go north"])
    assert action in ["look", "go north"]
    assert isinstance(reason, str)


def test_selector_no_actions():
    sel = InformedActionSelector()
    b = AgentBeliefs()
    action, reason = sel.select_action("room", b, [])
    assert action == "look"


def test_selector_with_mock_oracle():
    class MockOracle:
        def analyse_situation(self, obs, beliefs, actions):
            return SituationUnderstanding(
                recommended_action="stand up",
                action_reasoning="need to get out of bed",
                confidence=0.9,
            )

    sel = InformedActionSelector(oracle=MockOracle(), exploration_rate=0.0)
    b = AgentBeliefs()

    # Run multiple times — should usually pick "stand up"
    stand_count = sum(
        1 for _ in range(50)
        if sel.select_action("bedroom", b, ["stand up", "sleep", "look"])[0] == "stand up"
    )
    assert stand_count > 25  # Should strongly prefer "stand up"


def test_selector_uses_dynamics():
    dm = DynamicsModel()
    s = _make_state(1, world_hash="room")
    s_good = _make_state(2, world_hash="good")

    for _ in range(20):
        dm.update(s, "win", s_good, 10.0)

    sel = InformedActionSelector(dynamics=dm, exploration_rate=0.0)
    b = AgentBeliefs()
    b._location_id = 1
    b._world_hash = "room"

    win_count = sum(
        1 for _ in range(50)
        if sel.select_action("room", b, ["win", "lose"])[0] == "win"
    )
    assert win_count > 30


def test_selector_blocker_matching():
    sel = InformedActionSelector()
    assert sel._action_addresses_blocker("stand up", "need to stand first")
    assert sel._action_addresses_blocker("open door", "door is blocking the way")
    assert not sel._action_addresses_blocker("look", "need to stand first")


# ---------------------------------------------------------------------------
# BayesianIFAgent (v3)
# ---------------------------------------------------------------------------

def test_agent_choose_and_observe():
    agent = BayesianIFAgent(exploration_rate=0.5)
    agent.update_beliefs_from_ground_truth("bedroom", 1, [], "h0")
    action = agent.choose_action("You wake up.", ["stand up", "sleep", "look"])
    assert action in ["stand up", "sleep", "look"]

    agent.update_beliefs_from_ground_truth("bedroom", 1, [], "h1")
    agent.observe_outcome("You stand up.", 0.0, 1)


def test_agent_beliefs_updated_from_ground_truth():
    agent = BayesianIFAgent()
    agent.update_beliefs_from_ground_truth("kitchen", 5, ["knife", "plate"], "abc")
    assert agent.beliefs.location == "kitchen"
    assert agent.beliefs._location_id == 5
    assert "knife" in agent.beliefs.inventory
    assert agent.beliefs._world_hash == "abc"


def test_agent_reset_episode_keeps_goal():
    agent = BayesianIFAgent()
    agent.beliefs.overall_goal = "escape the house"
    agent.beliefs.accomplished = ["got dressed"]
    agent.reset_episode()
    assert agent.beliefs.overall_goal == "escape the house"
    assert agent.beliefs.accomplished == []  # Reset


def test_agent_action_history_tracked():
    agent = BayesianIFAgent()
    agent.update_beliefs_from_ground_truth("room", 1, [], "h0")
    agent.choose_action("Room.", ["look", "go north"])
    assert len(agent.beliefs.action_history) == 1


def test_agent_get_statistics():
    agent = BayesianIFAgent()
    stats = agent.get_statistics()
    assert "episode_count" in stats
    assert "transitions_learned" in stats
    assert "dynamics_history_size" in stats
    assert "overall_goal" in stats
    assert "reliability" in stats


def test_agent_dynamics_updated_after_observe():
    agent = BayesianIFAgent()
    agent.update_beliefs_from_ground_truth("room1", 1, [], "h0")
    agent.choose_action("Room 1.", ["go north"])

    agent.update_beliefs_from_ground_truth("room2", 2, [], "h1")
    agent.observe_outcome("Room 2.", 1.0, 1)

    assert len(agent.dynamics.history) >= 1
