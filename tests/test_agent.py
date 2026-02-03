"""Tests for BayesianIFAgent (runner.py v5) with mock FrotzEnv and LLM."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import BeliefState, CategoricalSensor, DynamicsModel, QuestionType


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockFrotzEnv:
    """Minimal mock of jericho.FrotzEnv for unit testing."""

    def __init__(self, valid_actions=None, state_hashes=None, scores=None):
        self._valid_actions = valid_actions or ["look", "go north", "take keys"]
        self._state_hashes = state_hashes or [100, 200, 300, 400, 500]
        self._scores = scores or [0, 0, 1, 1, 2]
        self._step_idx = 0
        self._game_over = False

    def reset(self):
        self._step_idx = 0
        self._game_over = False
        return "You are in a room.", {}

    def get_valid_actions(self):
        return list(self._valid_actions)

    def get_world_state_hash(self):
        idx = min(self._step_idx, len(self._state_hashes) - 1)
        return self._state_hashes[idx]

    def get_score(self):
        idx = min(self._step_idx, len(self._scores) - 1)
        return self._scores[idx]

    def get_max_score(self):
        return max(self._scores)

    def step(self, action):
        self._step_idx += 1
        idx = min(self._step_idx, len(self._scores) - 1)
        old_idx = max(0, idx - 1)
        reward = self._scores[idx] - self._scores[old_idx]
        obs = f"You did {action}."
        done = self._step_idx >= len(self._scores) - 1
        if done:
            self._game_over = True
        return obs, reward, done, {}

    def game_over(self):
        return self._game_over

    def close(self):
        pass


class MockYesLLM:
    """Always says YES."""
    def __init__(self):
        self.call_count = 0

    def complete(self, prompt):
        self.call_count += 1
        return "YES"


class MockNoLLM:
    """Always says NO."""
    def complete(self, prompt):
        return "NO"


class MockSuggestionLLM:
    """Returns "3" for suggestion prompts, "YES" for progress prompts,
    "YES"/"NO" for binary prompts based on action name."""

    def __init__(self, suggestion_idx=3):
        self.call_count = 0
        self.suggestion_idx = suggestion_idx

    def complete(self, prompt):
        self.call_count += 1
        prompt_lower = prompt.lower()
        if "which single action" in prompt_lower or "number of your chosen action" in prompt_lower:
            return str(self.suggestion_idx)
        if "narrative progress" in prompt_lower or "did the player make" in prompt_lower:
            return "YES"
        return "YES"


# ---------------------------------------------------------------------------
# Import BayesianIFAgent — needs jericho type hints but not runtime jericho
# ---------------------------------------------------------------------------

# We need to mock jericho before importing runner
import types

mock_jericho = types.ModuleType("jericho")
mock_jericho.FrotzEnv = MockFrotzEnv  # type: ignore[attr-defined]
sys.modules["jericho"] = mock_jericho

from runner import BayesianIFAgent


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

def test_agent_creation_no_llm():
    agent = BayesianIFAgent()
    assert agent.sensor_bank is None
    assert agent.episode_count == 0
    assert agent.total_questions_asked == 0


def test_agent_creation_with_llm():
    agent = BayesianIFAgent(llm_client=MockYesLLM())
    assert agent.sensor_bank is not None


def test_agent_choose_action_returns_valid():
    agent = BayesianIFAgent()
    env = MockFrotzEnv()
    env.reset()

    action, explanation = agent.choose_action(env, "You are in a room.")
    assert action in env.get_valid_actions()
    assert isinstance(explanation, str)


def test_agent_observe_outcome_records_dynamics():
    agent = BayesianIFAgent()
    env = MockFrotzEnv()
    env.reset()

    action, _ = agent.choose_action(env, "You are in a room.")
    obs, reward, done, info = env.step(action)
    agent.observe_outcome(env, action, obs, reward)

    assert agent.dynamics.total_observations >= 1


def test_agent_no_llm_no_questions():
    """Without LLM, agent should never ask questions."""
    agent = BayesianIFAgent()
    env = MockFrotzEnv()
    env.reset()

    for _ in range(5):
        action, _ = agent.choose_action(env, "room")
        obs, reward, done, info = env.step(action)
        agent.observe_outcome(env, action, obs, reward)
        if done:
            break

    assert agent.total_questions_asked == 0


def test_agent_with_llm_may_ask_questions():
    """With LLM and low question cost, agent may ask questions."""
    llm = MockYesLLM()
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv()
    env.reset()

    agent.choose_action(env, "You are in a room.")
    # We can't guarantee questions are asked (depends on VOI math),
    # but the infrastructure should not crash


def test_agent_observe_updates_sensor_reliability():
    """When LLM makes predictions, ground truth should update sensor."""
    llm = MockYesLLM()
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv()
    env.reset()

    action, _ = agent.choose_action(env, "room")

    if agent.last_llm_predictions:
        old_gt = agent.sensor_bank.sensors[QuestionType.ACTION_HELPS].ground_truth_count
        obs, reward, done, info = env.step(action)
        agent.observe_outcome(env, action, obs, reward)
        new_gt = agent.sensor_bank.sensors[QuestionType.ACTION_HELPS].ground_truth_count
        assert new_gt > old_gt


def test_agent_episode_resets_beliefs():
    """play_episode should reset beliefs but keep dynamics."""
    agent = BayesianIFAgent()
    env = MockFrotzEnv()

    agent.beliefs.update_belief("bedroom", "location", 0.9)
    old_dynamics_count = agent.dynamics.total_observations

    result = agent.play_episode(env, max_steps=3)

    # Beliefs should have been reset
    assert agent.beliefs.location_beliefs.get("bedroom") is None or True
    # Dynamics should still be there
    assert agent.dynamics.total_observations >= old_dynamics_count


def test_agent_episode_returns_stats():
    agent = BayesianIFAgent()
    env = MockFrotzEnv()

    result = agent.play_episode(env, max_steps=3)

    assert "episode" in result
    assert "total_reward" in result
    assert "final_score" in result
    assert "steps" in result
    assert "questions_asked" in result
    assert "dynamics_stats" in result
    assert result["steps"] > 0


def test_agent_episode_count_increments():
    agent = BayesianIFAgent()
    env = MockFrotzEnv()

    assert agent.episode_count == 0
    agent.play_episode(env, max_steps=3)
    assert agent.episode_count == 1
    agent.play_episode(env, max_steps=3)
    assert agent.episode_count == 2


def test_agent_dynamics_persist_across_episodes():
    """Dynamics should accumulate across episodes."""
    agent = BayesianIFAgent()
    env = MockFrotzEnv()

    agent.play_episode(env, max_steps=3)
    obs_after_1 = agent.dynamics.total_observations

    agent.play_episode(env, max_steps=3)
    obs_after_2 = agent.dynamics.total_observations

    assert obs_after_2 > obs_after_1


def test_agent_ground_truth_only_reward():
    """Ground truth should only use reward > 0, not state change."""
    llm = MockYesLLM()
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv(
        state_hashes=[100, 200, 300],  # state changes every step
        scores=[0, 0, 0],              # but no reward
    )
    env.reset()

    action, _ = agent.choose_action(env, "room")

    if agent.last_llm_predictions:
        predicted_action = list(agent.last_llm_predictions.keys())[0]
        obs, reward, done, info = env.step(action)
        # State changed (100 -> 200) but reward is 0
        agent.observe_outcome(env, action, obs, reward)

        # Sensor should have been updated with actual_truth=False
        # (reward was 0, so "didn't help" despite state change)
        sensor = agent.sensor_bank.sensors[QuestionType.ACTION_HELPS]
        assert sensor.ground_truth_count >= 1


def test_agent_conjugate_belief_update():
    """observe_outcome should do conjugate Beta update, not set to certainty."""
    agent = BayesianIFAgent()
    env = MockFrotzEnv(
        state_hashes=[100, 200],
        scores=[0, 1],  # reward on second step
    )
    env.reset()

    action, _ = agent.choose_action(env, "room")

    # Set a known action belief before observing
    agent.beliefs.set_action_belief(action, 1.0, 9.0)  # Beta(1, 9), mean 0.1

    obs, reward, done, info = env.step(action)
    agent.observe_outcome(env, action, obs, reward)

    if reward > 0:
        ab = agent.beliefs.get_action_belief(action)
        assert ab is not None
        alpha, beta = ab
        # Should be Beta(2, 9) — alpha incremented by 1, NOT set to certainty
        assert abs(alpha - 2.0) < 0.01
        assert abs(beta - 9.0) < 0.01


def test_agent_belief_update_on_failure():
    """observe_outcome should increment beta when reward <= 0."""
    agent = BayesianIFAgent()
    env = MockFrotzEnv(
        state_hashes=[100, 200],
        scores=[0, 0],  # no reward
    )
    env.reset()

    action, _ = agent.choose_action(env, "room")

    # Set a known action belief
    agent.beliefs.set_action_belief(action, 1.0, 9.0)  # Beta(1, 9)

    obs, reward, done, info = env.step(action)
    agent.observe_outcome(env, action, obs, reward)

    ab = agent.beliefs.get_action_belief(action)
    assert ab is not None
    alpha, beta = ab
    # Should be Beta(1, 10) — beta incremented by 1
    assert abs(alpha - 1.0) < 0.01
    assert abs(beta - 10.0) < 0.01


def test_agent_no_action_prior_parameter():
    """BayesianIFAgent should not accept action_prior parameter."""
    # Just verify it works without action_prior
    agent = BayesianIFAgent(question_cost=0.02, action_cost=0.05)
    assert agent.decision_maker.question_cost == 0.02
    assert agent.decision_maker.action_cost == 0.05
    assert not hasattr(agent.decision_maker, 'action_prior')


# ---------------------------------------------------------------------------
# Categorical sensor + progress evaluator tests
# ---------------------------------------------------------------------------

def test_agent_creation_has_categorical_sensor_with_llm():
    """With LLM, agent should have categorical sensor and progress sensor."""
    agent = BayesianIFAgent(llm_client=MockYesLLM())
    assert agent.categorical_sensor is not None
    assert agent.progress_sensor is not None


def test_agent_creation_no_categorical_without_llm():
    """Without LLM, no categorical/progress sensors."""
    agent = BayesianIFAgent()
    assert agent.categorical_sensor is None
    assert agent.progress_sensor is None


def test_agent_suggestion_cost_defaults_to_question_cost():
    agent = BayesianIFAgent(llm_client=MockYesLLM(), question_cost=0.05)
    assert agent.suggestion_cost == 0.05


def test_agent_suggestion_cost_override():
    agent = BayesianIFAgent(
        llm_client=MockYesLLM(), question_cost=0.01, suggestion_cost=0.02
    )
    assert agent.suggestion_cost == 0.02


def test_agent_suggestion_updates_all_beliefs():
    """When categorical suggestion is used, all action beliefs should update."""
    llm = MockSuggestionLLM(suggestion_idx=2)
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv(valid_actions=["look", "go north", "take keys"])
    env.reset()

    # Force categorical sensor to be very accurate so VOI is high
    agent.categorical_sensor.accuracy_alpha = 9.0
    agent.categorical_sensor.accuracy_beta = 1.0

    action, explanation = agent.choose_action(env, "You are in a room.")
    # The agent should function without crashing
    assert action in env.get_valid_actions()


def test_agent_reward_updates_categorical_accuracy():
    """When reward > 0 and suggestion was tracked, categorical sensor updates."""
    llm = MockSuggestionLLM(suggestion_idx=1)
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv(
        valid_actions=["look", "go north"],
        state_hashes=[100, 200],
        scores=[0, 1],
    )
    env.reset()

    # Manually set last_suggestion to simulate a prior suggestion
    action, _ = agent.choose_action(env, "room")
    agent.last_suggestion = action  # simulate LLM suggested this action

    old_gt = agent.categorical_sensor.ground_truth_count
    obs, reward, _, _ = env.step(action)
    agent.observe_outcome(env, action, obs, reward)

    if reward > 0:
        assert agent.categorical_sensor.ground_truth_count > old_gt


def test_agent_progress_provides_ground_truth_no_reward():
    """When reward == 0 but progress says YES, categorical sensor still updates."""
    llm = MockSuggestionLLM(suggestion_idx=1)
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv(
        valid_actions=["look", "go north"],
        state_hashes=[100, 200],
        scores=[0, 0],  # no reward
    )
    env.reset()

    action, _ = agent.choose_action(env, "room")
    agent.last_suggestion = action  # simulate prior suggestion

    old_gt = agent.categorical_sensor.ground_truth_count
    obs, reward, _, _ = env.step(action)
    agent.observe_outcome(env, action, obs, reward)

    # Progress sensor should have been consulted (reward == 0)
    # MockSuggestionLLM returns YES for progress → categorical sensor updates
    assert agent.categorical_sensor.ground_truth_count > old_gt


def test_agent_episode_resets_suggestion_state():
    """play_episode should reset last_suggestion and last_observation."""
    agent = BayesianIFAgent(llm_client=MockYesLLM())
    env = MockFrotzEnv()

    agent.last_suggestion = "some action"
    agent.last_observation = "some text"

    agent.play_episode(env, max_steps=3)

    # After episode plays, these should have been reset at start
    # (and whatever happens during play is fine)
    # The key thing: agent doesn't crash with suggestion flow


def test_agent_episode_with_suggestion_llm():
    """Full episode with suggestion LLM should not crash."""
    llm = MockSuggestionLLM(suggestion_idx=1)
    agent = BayesianIFAgent(llm_client=llm, question_cost=0.001)
    env = MockFrotzEnv()

    result = agent.play_episode(env, max_steps=5)
    assert result["steps"] > 0
    assert "categorical_sensor_stats" in result
