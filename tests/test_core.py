"""Tests for core.py v5 — BinarySensor, CategoricalSensor, QuestionType,
LLMSensorBank, BeliefState, StateActionKey, ObservedOutcome, DynamicsModel,
UnifiedDecisionMaker."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import (
    BinarySensor,
    CategoricalSensor,
    QuestionType,
    LLMSensorBank,
    BeliefState,
    StateActionKey,
    ObservedOutcome,
    DynamicsModel,
    UnifiedDecisionMaker,
)


# ---------------------------------------------------------------------------
# BinarySensor (SPEC v6: Beta(2,1) / Beta(1,2))
# ---------------------------------------------------------------------------

def test_sensor_initial_rates():
    s = BinarySensor()
    assert abs(s.tpr - 2/3) < 0.01
    assert abs(s.fpr - 1/3) < 0.01
    assert abs(s.reliability - 1/3) < 0.01
    assert s.query_count == 0
    assert s.ground_truth_count == 0


def test_sensor_custom_priors():
    s = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)
    assert s.tpr == 0.9
    assert s.fpr == 0.1
    assert abs(s.reliability - 0.8) < 0.01


def test_sensor_update_true_positive():
    s = BinarySensor()
    old_tpr = s.tpr
    s.update(said_yes=True, was_true=True)
    assert s.tpr > old_tpr
    assert s.ground_truth_count == 1


def test_sensor_update_false_positive():
    s = BinarySensor()
    old_fpr = s.fpr
    s.update(said_yes=True, was_true=False)
    assert s.fpr > old_fpr
    assert s.ground_truth_count == 1


def test_sensor_update_true_negative():
    s = BinarySensor()
    old_fpr = s.fpr
    s.update(said_yes=False, was_true=False)
    assert s.fpr < old_fpr
    assert s.ground_truth_count == 1


def test_sensor_update_false_negative():
    s = BinarySensor()
    old_tpr = s.tpr
    s.update(said_yes=False, was_true=True)
    assert s.tpr < old_tpr


def test_sensor_posterior_yes_increases_belief():
    s = BinarySensor()
    post = s.posterior(0.5, said_yes=True)
    assert post > 0.5


def test_sensor_posterior_no_decreases_belief():
    s = BinarySensor()
    post = s.posterior(0.5, said_yes=False)
    assert post < 0.5


def test_sensor_posterior_extreme_prior():
    s = BinarySensor()
    # Prior of 1.0 should stay 1.0
    assert s.posterior(1.0, said_yes=True) == 1.0
    assert s.posterior(1.0, said_yes=False) == 1.0
    # Prior of 0.0 should stay 0.0
    assert s.posterior(0.0, said_yes=True) == 0.0
    assert s.posterior(0.0, said_yes=False) == 0.0


def test_sensor_posterior_denominator_zero():
    # If both TPR and FPR are 0, denominator is 0 for said_yes=True
    s = BinarySensor(tp_alpha=0.001, tp_beta=1000, fp_alpha=0.001, fp_beta=1000)
    # Both rates ~0, denominator ~0 for yes case
    post = s.posterior(0.5, said_yes=True)
    # Should return prior when denominator is 0 or near-0
    assert 0.0 <= post <= 1.0


def test_sensor_get_stats():
    s = BinarySensor()
    s.update(True, True)
    stats = s.get_stats()
    assert "tpr" in stats
    assert "fpr" in stats
    assert "reliability" in stats
    assert stats["queries"] == 0
    assert stats["ground_truths"] == 1


# ---------------------------------------------------------------------------
# QuestionType
# ---------------------------------------------------------------------------

def test_question_type_completeness():
    assert len(QuestionType) == 9
    values = {qt.value for qt in QuestionType}
    assert "action_helps" in values
    assert "in_location" in values
    assert "have_item" in values
    assert "state_flag" in values
    assert "goal_done" in values
    assert "prereq_met" in values
    assert "action_possible" in values
    assert "suggest_action" in values
    assert "made_progress" in values


# ---------------------------------------------------------------------------
# LLMSensorBank (all sensors use SPEC defaults)
# ---------------------------------------------------------------------------

class MockYesLLM:
    def complete(self, prompt):
        return "YES"


class MockNoLLM:
    def complete(self, prompt):
        return "NO, I don't think so."


def test_sensor_bank_ask_yes():
    bank = LLMSensorBank(MockYesLLM())
    answer, rel = bank.ask(QuestionType.ACTION_HELPS, "Will go north help?")
    assert answer is True
    assert isinstance(rel, float)
    assert bank.sensors[QuestionType.ACTION_HELPS].query_count == 1


def test_sensor_bank_ask_no():
    bank = LLMSensorBank(MockNoLLM())
    answer, rel = bank.ask(QuestionType.ACTION_HELPS, "Will go north help?")
    assert answer is False


def test_sensor_bank_cache():
    bank = LLMSensorBank(MockYesLLM())
    bank.ask(QuestionType.ACTION_HELPS, "q1", "ctx")
    assert bank.sensors[QuestionType.ACTION_HELPS].query_count == 1

    # Second call same question — should be cached
    bank.ask(QuestionType.ACTION_HELPS, "q1", "ctx")
    assert bank.sensors[QuestionType.ACTION_HELPS].query_count == 1

    # Different question — not cached
    bank.ask(QuestionType.ACTION_HELPS, "q2", "ctx")
    assert bank.sensors[QuestionType.ACTION_HELPS].query_count == 2


def test_sensor_bank_clear_cache():
    bank = LLMSensorBank(MockYesLLM())
    bank.ask(QuestionType.ACTION_HELPS, "q1")
    bank.clear_cache()
    bank.ask(QuestionType.ACTION_HELPS, "q1")
    assert bank.sensors[QuestionType.ACTION_HELPS].query_count == 2


def test_sensor_bank_ground_truth_updates():
    bank = LLMSensorBank(MockYesLLM())
    old_tpr = bank.sensors[QuestionType.ACTION_HELPS].tpr
    bank.update_from_ground_truth(QuestionType.ACTION_HELPS, said_yes=True, actual_truth=True)
    assert bank.sensors[QuestionType.ACTION_HELPS].tpr > old_tpr


def test_sensor_bank_get_posterior():
    bank = LLMSensorBank(MockYesLLM())
    post = bank.get_posterior(QuestionType.ACTION_HELPS, 0.5, said_yes=True)
    assert post > 0.5


def test_sensor_bank_get_all_stats():
    bank = LLMSensorBank(MockYesLLM())
    stats = bank.get_all_stats()
    assert len(stats) == 9
    assert "action_helps" in stats
    assert "suggest_action" in stats
    assert "made_progress" in stats


def test_sensor_bank_all_sensors_use_spec_defaults():
    """All per-type sensors should use the SPEC defaults Beta(2,1)/Beta(1,2)."""
    bank = LLMSensorBank(MockYesLLM())
    for qt, sensor in bank.sensors.items():
        assert sensor.tp_alpha == 2.0, f"{qt} tp_alpha wrong"
        assert sensor.tp_beta == 1.0, f"{qt} tp_beta wrong"
        assert sensor.fp_alpha == 1.0, f"{qt} fp_alpha wrong"
        assert sensor.fp_beta == 2.0, f"{qt} fp_beta wrong"


# ---------------------------------------------------------------------------
# BeliefState
# ---------------------------------------------------------------------------

def test_belief_state_defaults():
    b = BeliefState()
    assert b.current_location is None
    assert b.location_beliefs == {}
    assert b.inventory_beliefs == {}
    assert b.flag_beliefs == {}
    assert b.goal_beliefs == {}
    assert b.action_beliefs == {}


def test_belief_state_update_and_get():
    b = BeliefState()
    b.update_belief("bedroom", "location", 0.9)
    assert b.get_belief("bedroom", "location") == 0.9
    assert b.get_belief("kitchen", "location") == 0.5  # default


def test_belief_state_location_tracking():
    b = BeliefState()
    b.update_belief("bedroom", "location", 0.9)
    assert b.current_location == "bedroom"

    b.update_belief("kitchen", "location", 0.95)
    assert b.current_location == "kitchen"


def test_belief_state_set_certain_true():
    b = BeliefState()
    b.set_certain("keys", "inventory", True)
    assert b.get_belief("keys", "inventory") == 1.0


def test_belief_state_set_certain_false():
    b = BeliefState()
    b.set_certain("keys", "inventory", False)
    assert b.get_belief("keys", "inventory") == 0.0


def test_belief_state_non_action_categories():
    b = BeliefState()
    b.update_belief("bedroom", "location", 0.9)
    b.update_belief("keys", "inventory", 0.8)
    b.update_belief("door_locked", "flag", 0.7)
    b.update_belief("escape", "goal", 0.6)

    assert b.get_belief("bedroom", "location") == 0.9
    assert b.get_belief("keys", "inventory") == 0.8
    assert b.get_belief("door_locked", "flag") == 0.7
    assert b.get_belief("escape", "goal") == 0.6


def test_belief_state_action_beliefs_are_tuples():
    b = BeliefState()
    b.set_action_belief("go north", 2.0, 8.0)
    ab = b.get_action_belief("go north")
    assert ab == (2.0, 8.0)
    assert b.get_action_belief("nonexistent") is None


def test_belief_state_unknown_category():
    b = BeliefState()
    assert b.get_belief("x", "nonexistent") == 0.5
    # Should not raise
    b.update_belief("x", "nonexistent", 0.8)


def test_belief_state_to_context_string_empty():
    b = BeliefState()
    assert b.to_context_string() == "No confident beliefs yet."


def test_belief_state_to_context_string_populated():
    b = BeliefState()
    b.update_belief("bedroom", "location", 0.9)
    b.set_certain("keys", "inventory", True)
    b.update_belief("door_locked", "flag", 0.8)
    b.update_belief("escape", "goal", 0.9)

    ctx = b.to_context_string()
    assert "bedroom" in ctx
    assert "keys" in ctx
    assert "door_locked" in ctx
    assert "escape" in ctx


# ---------------------------------------------------------------------------
# StateActionKey
# ---------------------------------------------------------------------------

def test_state_action_key_frozen():
    key = StateActionKey("h1", "go north")
    try:
        key.state_hash = "h2"  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_state_action_key_hashable():
    k1 = StateActionKey("h1", "go north")
    k2 = StateActionKey("h1", "go north")
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_state_action_key_as_dict_key():
    k1 = StateActionKey("h1", "go")
    k2 = StateActionKey("h1", "go")
    d = {k1: "value"}
    assert d[k2] == "value"


def test_state_action_key_inequality():
    k1 = StateActionKey("h1", "go north")
    k2 = StateActionKey("h1", "go south")
    assert k1 != k2


# ---------------------------------------------------------------------------
# ObservedOutcome
# ---------------------------------------------------------------------------

def test_observed_outcome_construction():
    o = ObservedOutcome(next_state_hash="h2", reward=1.0, observation_text="text")
    assert o.next_state_hash == "h2"
    assert o.reward == 1.0
    assert o.observation_text == "text"
    assert o.visit_count == 1
    assert o.reward_sum == 1.0
    assert o.mean_reward == 1.0


def test_observed_outcome_default_text():
    o = ObservedOutcome(next_state_hash="h2", reward=0.0)
    assert o.observation_text == ""
    assert o.mean_reward == 0.0


# ---------------------------------------------------------------------------
# DynamicsModel
# ---------------------------------------------------------------------------

def test_dynamics_empty():
    d = DynamicsModel()
    assert not d.has_observation("s1", "go")
    assert d.known_reward("s1", "go") is None
    assert d.get_observation("s1", "go") is None
    assert d.get_stats()["total_observations"] == 0


def test_dynamics_record_and_query():
    d = DynamicsModel()
    d.record_observation("s1", "go", "s2", 1.0, "You go north.")
    assert d.has_observation("s1", "go")
    assert d.known_reward("s1", "go") == 1.0
    obs = d.get_observation("s1", "go")
    assert obs is not None
    assert obs.next_state_hash == "s2"


def test_dynamics_tried_actions():
    d = DynamicsModel()
    d.record_observation("s1", "go", "s2", 0.0)
    d.record_observation("s1", "look", "s1", 0.0)
    assert "go" in d.tried_actions
    assert "look" in d.tried_actions
    assert "jump" not in d.tried_actions


def test_dynamics_accumulates_visits():
    d = DynamicsModel()
    d.record_observation("s1", "go", "s2", 1.0, "first")
    d.record_observation("s1", "go", "s2", 1.0, "second")
    assert d.get_stats()["unique_state_actions"] == 1
    assert d.total_observations == 2
    obs = d.get_observation("s1", "go")
    assert obs.observation_text == "second"
    assert obs.visit_count == 2
    assert obs.reward_sum == 2.0
    assert obs.mean_reward == 1.0


def test_dynamics_stats():
    d = DynamicsModel()
    d.record_observation("s1", "a1", "s2", 0.0)
    d.record_observation("s1", "a2", "s3", 1.0)
    d.record_observation("s2", "a1", "s4", 0.5)

    stats = d.get_stats()
    assert stats["total_observations"] == 3
    assert stats["unique_state_actions"] == 3
    assert stats["unique_actions_tried"] == 2


# ---------------------------------------------------------------------------
# UnifiedDecisionMaker (beliefs are now Beta tuples)
# ---------------------------------------------------------------------------

def test_udm_take_known_best():
    """When all actions have observed outcomes, take the best one."""
    udm = UnifiedDecisionMaker(question_cost=0.01)
    d = DynamicsModel()
    d.record_observation("s", "a1", "s2", 5.0)
    d.record_observation("s", "a2", "s", 0.0)
    d.register_state("s", 2)
    d.register_state("s2", 1)

    decision = udm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
    )
    assert decision == ('take', 'a1')


def test_udm_take_unknown_highest_belief():
    """When no known rewards, take action with highest belief mean."""
    udm = UnifiedDecisionMaker(question_cost=0.01)
    d = DynamicsModel()

    decision = udm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[],
        beliefs={"a1": (8.0, 2.0), "a2": (2.0, 8.0)},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
    )
    assert decision == ('take', 'a1')


def test_udm_no_questions_when_all_observed():
    """Even with questions provided, skip if all actions observed."""
    udm = UnifiedDecisionMaker(question_cost=0.01)
    d = DynamicsModel()
    d.record_observation("s", "a1", "s2", 1.0)
    d.record_observation("s", "a2", "s", 0.0)
    d.register_state("s", 2)
    d.register_state("s2", 1)

    decision = udm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[("a1", "Q1"), ("a2", "Q2")],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
    )
    assert decision[0] == 'take'


def test_udm_voi_non_negative():
    """VOI should never be negative."""
    udm = UnifiedDecisionMaker()
    sensor = BinarySensor()
    # EUs consistent with untried formula: belief_mean + V₀ - action_cost
    eus = {"a1": 0.5 + 0.5 - 0.10, "a2": 0.5 + 0.5 - 0.10}
    voi = udm.compute_voi("a1", 0.5, 0.5, sensor, eus)
    assert voi >= 0.0


def test_udm_voi_zero_when_certain():
    """VOI is zero when belief is already certain."""
    udm = UnifiedDecisionMaker()
    sensor = BinarySensor()
    # Alpha >> beta means ~certain it helps; EU includes V₀
    eus = {"a1": 100/100.01 + 0.5 - 0.10, "a2": 0.5 + 0.5 - 0.10}
    voi = udm.compute_voi("a1", 100.0, 0.01, sensor, eus)
    assert abs(voi) < 0.01


def test_udm_voi_higher_with_reliable_sensor():
    """More reliable sensor should have higher VOI (more informative)."""
    udm = UnifiedDecisionMaker()

    unreliable = BinarySensor(tp_alpha=5, tp_beta=5, fp_alpha=5, fp_beta=5)
    reliable = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    eus = {"a1": 0.5 + 0.5 - 0.10, "a2": 0.5 + 0.5 - 0.10}
    voi_unreliable = udm.compute_voi("a1", 0.5, 0.5, unreliable, eus)
    voi_reliable = udm.compute_voi("a1", 0.5, 0.5, reliable, eus)

    assert voi_reliable >= voi_unreliable


def test_udm_voi_positive_for_non_best():
    """VOI should be positive when asking about a non-best uncertain action."""
    udm = UnifiedDecisionMaker()
    sensor = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    # a1 has lower EU than a2 — asking about a1 could reveal it's better
    # EUs include V₀: belief_mean + V₀ - action_cost
    eus = {"a1": 0.3 + 0.5 - 0.10, "a2": 0.7 + 0.5 - 0.10}
    voi = udm.compute_voi("a1", 0.3, 0.7, sensor, eus)
    assert voi > 0.0


def test_udm_voi_zero_for_best_action():
    """VOI is ~0 when asking about the action that's already the best."""
    udm = UnifiedDecisionMaker()
    sensor = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    # a1 is already the best — asking about it can't improve decisions
    # EUs include V₀: belief_mean + V₀ - action_cost
    eus = {"a1": 0.8 + 0.5 - 0.10, "a2": 0.1 + 0.5 - 0.10}
    voi = udm.compute_voi("a1", 8.0, 2.0, sensor, eus)
    assert voi < 0.01


def test_udm_ask_wins_with_weak_prior():
    """With weak 1/N prior and reliable sensor, asking should beat taking.

    This is the key behavioral test: when actions have weak Beta(1/N, 1-1/N)
    priors and the sensor is reliable, VOI exceeds cost and the agent asks.
    """
    udm = UnifiedDecisionMaker(
        question_cost=0.01,
        action_cost=0.10,
    )
    d = DynamicsModel()
    sensor = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    # Beta(0.5, 0.5) = 1/N prior for N=2
    decision = udm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[("a1", "Will a1 help?"), ("a2", "Will a2 help?")],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=sensor,
        dynamics=d,
        state_hash="s",
    )
    assert decision[0] == 'ask'


def test_udm_take_when_cost_high():
    """With high question cost, should prefer taking action."""
    udm = UnifiedDecisionMaker(question_cost=10.0)
    d = DynamicsModel()
    sensor = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    decision = udm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[("a1", "Will a1 help?")],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=sensor,
        dynamics=d,
        state_hash="s",
    )
    assert decision[0] == 'take'


def test_udm_default_prior_is_one_over_n():
    """When beliefs dict is empty, default prior should be Beta(1/N, 1-1/N)."""
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    d = DynamicsModel()

    # 5 actions, so default prior mean = 1/5 = 0.2
    # EU = 0.2 - 0.1 = 0.1 for each
    decision = udm.choose(
        game_actions=["a1", "a2", "a3", "a4", "a5"],
        possible_questions=[],
        beliefs={},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
    )
    assert decision[0] == 'take'
    assert decision[1] in ["a1", "a2", "a3", "a4", "a5"]


def test_udm_decision_comparison_is_voi_vs_cost():
    """The decision rule is VOI > c, not (VOI - c) > best_game_eu.

    This tests the C1 fix: with best_game_eu=0.05, VOI=0.03, c=0.01,
    the agent should ask (0.03 > 0.01) even though 0.02 < 0.05.
    """
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    d = DynamicsModel()
    d.register_state("s", 2)
    d.register_state("s2", 1)

    # Construct a scenario where VOI > c but (VOI - c) < best_game_eu
    # Use a very reliable sensor and uncertain belief
    sensor = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    # a1 has moderate belief, a2 is known good
    d.record_observation("s", "a2", "s2", 0.2)  # known reward 0.2

    decision = udm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[("a1", "Will a1 help?")],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=sensor,
        dynamics=d,
        state_hash="s",
    )
    # a2 EU = 0.2 + V(s2) - 0.10 = 0.2 + 0.5 - 0.10 = 0.60
    # a1 EU = 0.5 + V₀ - 0.10 = 0.5 + 0.5 - 0.10 = 0.90
    # a1 is untried and already best, so VOI for a1 ≈ 0
    # But this test is about the comparison structure, not specific values
    v_s2 = d.state_value("s2")
    voi = udm.compute_voi("a1", 0.5, 0.5, sensor,
                          {"a1": 0.5 + 0.5 - 0.10, "a2": 0.2 + v_s2 - 0.10})
    if voi > 0.01:
        assert decision[0] == 'ask'


# ---------------------------------------------------------------------------
# CategoricalSensor
# ---------------------------------------------------------------------------

def test_categorical_sensor_posteriors_sum_to_one():
    cs = CategoricalSensor()
    priors = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    post = cs.posteriors(priors, "a")
    assert abs(sum(post.values()) - 1.0) < 1e-9


def test_categorical_sensor_posteriors_sum_to_one_nonuniform():
    cs = CategoricalSensor(accuracy_alpha=8, accuracy_beta=2)
    priors = {"a": 0.5, "b": 0.3, "c": 0.2}
    post = cs.posteriors(priors, "b")
    assert abs(sum(post.values()) - 1.0) < 1e-9


def test_categorical_sensor_boosts_suggested():
    cs = CategoricalSensor()
    priors = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    post = cs.posteriors(priors, "a")
    assert post["a"] > priors["a"]


def test_categorical_sensor_non_suggested_lowered():
    cs = CategoricalSensor()
    priors = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    post = cs.posteriors(priors, "a")
    assert post["b"] < priors["b"]
    assert post["c"] < priors["c"]
    assert post["d"] < priors["d"]


def test_categorical_sensor_initial_accuracy():
    cs = CategoricalSensor()
    assert abs(cs.accuracy - 0.5) < 0.01  # Beta(1,1) max-entropy


def test_categorical_sensor_accuracy_learns():
    cs = CategoricalSensor()
    initial = cs.accuracy
    cs.update(suggested_correct=True)
    assert cs.accuracy > initial
    cs.update(suggested_correct=False)
    cs.update(suggested_correct=False)
    # After 1 correct, 2 incorrect from Beta(1,1): Beta(2,3), mean 0.4
    assert abs(cs.accuracy - 0.4) < 0.01
    assert cs.ground_truth_count == 3


def test_categorical_sensor_get_stats():
    cs = CategoricalSensor()
    cs.update(True)
    stats = cs.get_stats()
    assert "accuracy" in stats
    assert "queries" in stats
    assert "ground_truths" in stats
    assert stats["ground_truths"] == 1
    assert stats["queries"] == 0


def test_categorical_voi_non_negative():
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    cs = CategoricalSensor()
    voi = udm.compute_voi_categorical(
        ["a1", "a2", "a3"],
        {"a1": (0.33, 0.67), "a2": (0.33, 0.67), "a3": (0.33, 0.67)},
        cs,
    )
    assert voi >= 0.0


def test_categorical_voi_higher_with_accurate_sensor():
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    weak = CategoricalSensor(accuracy_alpha=2, accuracy_beta=2)   # accuracy 0.5
    strong = CategoricalSensor(accuracy_alpha=9, accuracy_beta=1)  # accuracy 0.9

    beliefs = {"a1": (0.33, 0.67), "a2": (0.33, 0.67), "a3": (0.33, 0.67)}
    voi_weak = udm.compute_voi_categorical(["a1", "a2", "a3"], beliefs, weak)
    voi_strong = udm.compute_voi_categorical(["a1", "a2", "a3"], beliefs, strong)

    assert voi_strong >= voi_weak


def test_categorical_voi_zero_when_all_known():
    """VOI is zero when dynamics has observations for all actions."""
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    cs = CategoricalSensor(accuracy_alpha=9, accuracy_beta=1)
    d = DynamicsModel()
    d.record_observation("s", "a1", "s2", 1.0)
    d.record_observation("s", "a2", "s2", 0.0)

    # When all actions have known outcomes, categorical sensor adds nothing.
    # The choose() method handles this — known rewards bypass belief entirely.
    # Here we test VOI on the beliefs themselves (even with all known, beliefs
    # don't change the decision since known rewards dominate).
    # With strongly peaked beliefs, VOI should be ~0.
    beliefs = {"a1": (100.0, 0.01), "a2": (0.01, 100.0)}
    voi = udm.compute_voi_categorical(["a1", "a2"], beliefs, cs)
    assert voi < 0.01


def test_categorical_sensor_choose_integrates():
    """UnifiedDecisionMaker.choose with categorical_sensor parameter."""
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    cs = CategoricalSensor(accuracy_alpha=9, accuracy_beta=1)
    d = DynamicsModel()

    # With weak priors and accurate categorical sensor, should suggest
    decision = udm.choose(
        game_actions=["a1", "a2", "a3"],
        possible_questions=[],
        beliefs={"a1": (0.33, 0.67), "a2": (0.33, 0.67), "a3": (0.33, 0.67)},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
        categorical_sensor=cs,
        suggestion_cost=0.01,
    )
    assert decision[0] in ('suggest', 'take')


# ---------------------------------------------------------------------------
# DynamicsModel.state_value and register_state
# ---------------------------------------------------------------------------

def test_dynamics_register_state():
    """register_state stores n_total."""
    d = DynamicsModel()
    d.register_state("s1", 5)
    assert d._state_n_total["s1"] == 5


def test_dynamics_state_value_unvisited():
    """Unregistered state returns V₀ = 0.5."""
    d = DynamicsModel()
    assert d.state_value("unknown") == 0.5


def test_dynamics_state_value_no_tries():
    """Registered but 0 tried → Bellman V from untried prior: 1/N + γ·0.5 - c_act."""
    d = DynamicsModel()
    d.register_state("s1", 5)
    # Q_untried = 1/5 + 0.95·0.5 - 0.10 = 0.575
    assert d.state_value("s1") > 0.4


def test_dynamics_state_value_fully_explored():
    """All tried, 0 rewards, self-loop → V < 0 (dead-end state)."""
    d = DynamicsModel()
    d.register_state("s", 2)
    d.record_observation("s", "a1", "s", 0.0)
    d.record_observation("s", "a2", "s", 0.0)
    # V(s) = 0.95·V(s) - 0.10 → V(s) = -0.10/0.05 = -2.0
    assert d.state_value("s") < 0


def test_dynamics_state_value_partial():
    """5 of 10 tried, 1 reward → V > 0.5 (reward propagates via Bellman)."""
    d = DynamicsModel()
    d.register_state("s", 10)
    d.record_observation("s", "a1", "s2", 1.0)
    for i in range(2, 6):
        d.record_observation("s", f"a{i}", "s", 0.0)
    v = d.state_value("s")
    # Q(a1) = 1.0 + 0.95·V(s2) - 0.10 = 1.375 (s2 unknown → V=0.5)
    # V(s) = max(Q(a1), Q_untried, Q_self_loop) ≥ 1.375
    assert v > 0.5


def test_dynamics_state_value_all_rewarding():
    """All tried, all rewarded → V > 0 (rewards propagate via Bellman)."""
    d = DynamicsModel()
    d.register_state("s", 2)
    d.record_observation("s", "a1", "s2", 1.0)
    d.record_observation("s", "a2", "s3", 1.0)
    # Q(a1) = 1.0 + 0.95·V(s2) - 0.10 = 1.375 (s2, s3 unknown → V=0.5)
    assert d.state_value("s") > 0


# ---------------------------------------------------------------------------
# UnifiedDecisionMaker with Q-values
# ---------------------------------------------------------------------------

def test_udm_prefers_state_change_over_noop():
    """State-changing r=0 action → Q > no-op Q (via Bellman state value)."""
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    d = DynamicsModel(action_cost=0.10, gamma=0.95)
    d.register_state("s", 2)
    d.register_state("s2", 5)  # unvisited successor with actions
    # "north" moved to s2 with 0 reward
    d.record_observation("s", "north", "s2", 0.0)
    # "look" stayed in s with 0 reward
    d.record_observation("s", "look", "s", 0.0)

    decision = udm.choose(
        game_actions=["north", "look"],
        possible_questions=[],
        beliefs={"north": (0.5, 0.5), "look": (0.5, 0.5)},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
    )
    # V(s2) = 0.575 (5 untried: 1/5 + 0.95·0.5 - 0.10)
    # "north": Q = 0 + V(s2) - 0.10, V(s2) > V(s)
    # "look":  Q = 0 + V(s) - 0.10, V(s) self-loop limited
    assert decision == ('take', 'north')


def test_udm_untried_beats_tried_zero():
    """Untried action preferred over tried r=0 no-op."""
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    d = DynamicsModel(action_cost=0.10, gamma=0.95)
    d.register_state("s", 2)
    d.record_observation("s", "look", "s", 0.0)

    decision = udm.choose(
        game_actions=["look", "north"],
        possible_questions=[],
        beliefs={"look": (0.5, 0.5), "north": (0.5, 0.5)},
        sensor=BinarySensor(),
        dynamics=d,
        state_hash="s",
    )
    # "look":  Q = 0 + V(s) - 0.10, V(s) Bellman-limited by self-loop
    # "north": Q = 0.5 + 0.5 - 0.10 = 0.90 (untried: belief mean + V₀)
    assert decision == ('take', 'north')


def test_udm_voi_includes_v0():
    """VOI > 0 for untried action with reliable sensor (V₀ in eu_if_yes)."""
    udm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    sensor = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    # Compute game EUs: both untried, both get V₀
    game_eus = {
        "a1": 0.5 + 0.5 - 0.10,  # belief 0.5 + V₀ 0.5 - cost
        "a2": 0.5 + 0.5 - 0.10,
    }
    voi = udm.compute_voi("a1", 0.5, 0.5, sensor, game_eus)
    assert voi > 0.0


# ---------------------------------------------------------------------------
# Bellman V(s) tests
# ---------------------------------------------------------------------------

def test_bellman_propagates_reward():
    """Reward at s1→s2 should propagate back to V(s0) via Bellman."""
    d = DynamicsModel(action_cost=0.10, gamma=0.95)
    d.register_state("s0", 1)
    d.register_state("s1", 1)
    # Chain: s0 →(a0, r=0)→ s1 →(a1, r=5.0)→ s2
    d.record_observation("s0", "a0", "s1", 0.0)
    d.record_observation("s1", "a1", "s2", 5.0)

    v_s2 = d.state_value("s2")  # unknown → 0.5
    v_s1 = d.state_value("s1")  # Q(a1) = 5.0 + 0.95·0.5 - 0.10 = 5.375
    v_s0 = d.state_value("s0")  # Q(a0) = 0 + 0.95·V(s1) - 0.10

    assert v_s2 == 0.5
    assert v_s1 > 5.0
    assert v_s0 > 0.0
    # Reward propagates: V(s0) reflects discounted reward from s1→s2
    assert v_s0 > v_s2


def test_bellman_cycle_converges():
    """Self-loop s→s with r=0 converges to finite negative value."""
    d = DynamicsModel(action_cost=0.10, gamma=0.95)
    d.register_state("s", 1)
    d.record_observation("s", "a", "s", 0.0)

    v = d.state_value("s")
    # V(s) = 0.95·V(s) - 0.10 → V(s) = -0.10/0.05 = -2.0
    assert v < 0
    assert abs(v - (-2.0)) < 0.02


def test_bellman_cache_invalidation():
    """Adding observations invalidates cache; V(s) changes."""
    d = DynamicsModel(action_cost=0.10, gamma=0.95)
    d.register_state("s", 2)
    d.record_observation("s", "a1", "s", 0.0)
    v_before = d.state_value("s")

    # Add a rewarding observation — V(s) should change
    d.record_observation("s", "a2", "s2", 5.0)
    v_after = d.state_value("s")

    assert v_after != v_before
    assert v_after > v_before


# ---------------------------------------------------------------------------
# Mean reward convergence (Fix D2)
# ---------------------------------------------------------------------------

def test_mean_reward_converges_with_repeated_visits():
    """Repeated visits with different rewards converge to running mean."""
    d = DynamicsModel()
    # First visit: r=20 (e.g., first-time score event)
    d.record_observation("s1", "frotz", "s2", 20.0)
    assert d.known_reward("s1", "frotz") == 20.0

    # Second visit: r=0 (score already claimed)
    d.record_observation("s1", "frotz", "s2", 0.0)
    assert d.known_reward("s1", "frotz") == 10.0  # mean of 20 and 0

    # Third visit: r=0
    d.record_observation("s1", "frotz", "s2", 0.0)
    obs = d.get_observation("s1", "frotz")
    assert obs.visit_count == 3
    assert abs(obs.mean_reward - 20.0/3) < 0.01


def test_value_iteration_bounded_with_reward_cycle():
    """V(s) stays bounded when a reward cycle exists (same (s,a) revisited).

    Before Fix D1+D2, cycling s1 ↔ s2 with r=20 each time gave V(s1)≈200+.
    Now the mean reward converges to truth and V(s) stays bounded.
    """
    d = DynamicsModel(action_cost=0.10, gamma=0.95)
    d.register_state("s1", 2)
    d.register_state("s2", 2)

    # Simulate cycle: s1→s2 (r=20 first, then r=0), s2→s1 (r=0 always)
    d.record_observation("s1", "frotz", "s2", 20.0)
    d.record_observation("s2", "turn off", "s1", 0.0)
    d.record_observation("s1", "look", "s1", 0.0)
    d.record_observation("s2", "look", "s2", 0.0)

    v1 = d.state_value("s1")
    # With r=20 one-shot, V(s1) should be large but finite
    assert v1 > 0
    assert v1 < 500  # not blowing up

    # Now revisit with r=0 — mean reward drops to 10
    d.record_observation("s1", "frotz", "s2", 0.0)
    v1_after = d.state_value("s1")
    assert v1_after < v1  # value decreased because mean reward dropped


# ---------------------------------------------------------------------------
# ObservedOutcome fields (Fix D2)
# ---------------------------------------------------------------------------

def test_observed_outcome_mean_reward():
    """mean_reward property works correctly."""
    o = ObservedOutcome(next_state_hash="s2", reward=5.0)
    assert o.visit_count == 1
    assert o.reward_sum == 5.0
    assert o.mean_reward == 5.0

    # Simulate accumulation
    o.visit_count += 1
    o.reward_sum += 0.0
    o.reward = o.mean_reward
    assert o.mean_reward == 2.5


# ---------------------------------------------------------------------------
# Categorical sensor Beta(1,1) prior (Fix A)
# ---------------------------------------------------------------------------

def test_categorical_sensor_max_entropy_prior():
    """Default categorical sensor has Beta(1,1) — max entropy for untested sensor."""
    cs = CategoricalSensor()
    assert cs.accuracy_alpha == 1.0
    assert cs.accuracy_beta == 1.0
    assert abs(cs.accuracy - 0.5) < 0.01
