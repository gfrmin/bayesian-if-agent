"""
Bayesian Interactive Fiction Agent — v5

Binary sensor bank + categorical suggestion sensor + unified expected utility
maximisation.

Core insight: instead of asking the LLM for rich structured analysis, ask it
simple questions and learn each question type's reliability via Beta distributions.

Sensor types:
  - BinarySensor: yes/no questions with learned TPR/FPR
  - CategoricalSensor: "which action?" with learned scalar accuracy

All components are stdlib-only. LLM client is injected via duck typing
(any object with a .complete(prompt) -> str method).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Set
from enum import Enum


# =============================================================================
# BINARY SENSOR
# =============================================================================

@dataclass
class BinarySensor:
    """
    A yes/no sensor with learned reliability.

    Models P(sensor_output | truth) as Beta distributions.
    """

    # True positive: P(says Yes | actually Yes)
    tp_alpha: float = 2.0   # Prior: Beta(2,1), mean ~0.67 TPR
    tp_beta: float = 1.0

    # False positive: P(says Yes | actually No)
    fp_alpha: float = 1.0   # Prior: Beta(1,2), mean ~0.33 FPR
    fp_beta: float = 2.0

    # Query count (for diagnostics)
    query_count: int = 0
    ground_truth_count: int = 0

    @property
    def tpr(self) -> float:
        """Expected true positive rate."""
        return self.tp_alpha / (self.tp_alpha + self.tp_beta)

    @property
    def fpr(self) -> float:
        """Expected false positive rate."""
        return self.fp_alpha / (self.fp_alpha + self.fp_beta)

    @property
    def reliability(self) -> float:
        """Overall reliability score (TPR - FPR). Ranges from -1 to 1."""
        return self.tpr - self.fpr

    def update(self, said_yes: bool, was_true: bool):
        """Update reliability from ground truth."""
        self.ground_truth_count += 1

        if was_true:
            if said_yes:
                self.tp_alpha += 1  # True positive
            else:
                self.tp_beta += 1   # False negative
        else:
            if said_yes:
                self.fp_alpha += 1  # False positive
            else:
                self.fp_beta += 1   # True negative

    def posterior(self, prior: float, said_yes: bool) -> float:
        """
        P(true | LLM response) via Bayes' rule.

        Args:
            prior: P(true) before observing LLM response
            said_yes: Whether LLM said yes

        Returns:
            P(true | LLM response)
        """
        if said_yes:
            p_yes_if_true = self.tpr
            p_yes_if_false = self.fpr

            numerator = p_yes_if_true * prior
            denominator = p_yes_if_true * prior + p_yes_if_false * (1 - prior)
        else:
            p_no_if_true = 1 - self.tpr
            p_no_if_false = 1 - self.fpr

            numerator = p_no_if_true * prior
            denominator = p_no_if_true * prior + p_no_if_false * (1 - prior)

        if denominator == 0:
            return prior

        return numerator / denominator

    def get_stats(self) -> dict:
        return {
            "tpr": self.tpr,
            "fpr": self.fpr,
            "reliability": self.reliability,
            "queries": self.query_count,
            "ground_truths": self.ground_truth_count,
        }


# =============================================================================
# CATEGORICAL SENSOR
# =============================================================================

@dataclass
class CategoricalSensor:
    """
    Categorical sensor: picks one of N options, learned scalar accuracy.

    Models P(LLM picks correct action | truth) as a scalar accuracy
    with Beta prior. When the LLM suggests action i, the posterior for
    each action j is computed via Bayes' rule assuming uniform error
    over wrong actions.
    """

    accuracy_alpha: float = 1.0  # Beta(1,1), max-entropy for untested sensor
    accuracy_beta: float = 1.0

    query_count: int = 0
    ground_truth_count: int = 0

    @property
    def accuracy(self) -> float:
        """Expected accuracy (mean of Beta distribution)."""
        return self.accuracy_alpha / (self.accuracy_alpha + self.accuracy_beta)

    def update(self, suggested_correct: bool):
        """Update accuracy from ground truth."""
        self.ground_truth_count += 1
        if suggested_correct:
            self.accuracy_alpha += 1
        else:
            self.accuracy_beta += 1

    def posteriors(self, priors: Dict[str, float], suggested: str) -> Dict[str, float]:
        """
        P(action_j correct | LLM suggests action_i) via Bayes' rule.

        Likelihood model:
          P(LLM suggests i | action i is correct) = accuracy
          P(LLM suggests i | action j is correct, j != i) = (1 - accuracy) / (N - 1)
        """
        n = len(priors)
        acc = self.accuracy
        wrong_prob = (1.0 - acc) / max(n - 1, 1)
        unnormalized = {
            a: (acc if a == suggested else wrong_prob) * p
            for a, p in priors.items()
        }
        total = sum(unnormalized.values())
        return {a: p / total for a, p in unnormalized.items()} if total > 0 else dict(priors)

    def get_stats(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "queries": self.query_count,
            "ground_truths": self.ground_truth_count,
        }


# =============================================================================
# QUESTION TYPES
# =============================================================================

class QuestionType(Enum):
    ACTION_HELPS = "action_helps"       # "Will action X make progress?"
    IN_LOCATION = "in_location"         # "Am I in the bedroom?"
    HAVE_ITEM = "have_item"             # "Do I have the keys?"
    STATE_FLAG = "state_flag"           # "Is the door locked?"
    GOAL_DONE = "goal_done"             # "Have I answered the phone?"
    PREREQ_MET = "prereq_met"           # "Can I go east?"
    ACTION_POSSIBLE = "action_possible" # "Is 'open door' available?"
    SUGGEST_ACTION = "suggest_action"   # "Which action should I take?" (categorical)
    MADE_PROGRESS = "made_progress"     # "Did that action make narrative progress?"


# =============================================================================
# LLM SENSOR BANK
# =============================================================================

class LLMSensorBank:
    """
    The LLM as a collection of binary sensors.

    Each question type has independently learned reliability.
    LLM client is injected — this class remains stdlib-only.
    """

    def __init__(self, llm_client):
        self.llm = llm_client

        self.sensors: Dict[QuestionType, BinarySensor] = {
            qt: BinarySensor() for qt in QuestionType
        }

        # Cache to avoid redundant queries within same turn
        self.query_cache: Dict[str, bool] = {}

    def ask(
        self,
        question_type: QuestionType,
        question: str,
        context: str = "",
    ) -> Tuple[bool, float]:
        """
        Ask a yes/no question.

        Returns: (answer, reliability)
        """
        cache_key = f"{question_type.value}:{question}"

        if cache_key in self.query_cache:
            return self.query_cache[cache_key], self.sensors[question_type].reliability

        answer = self._query_llm(question, context)

        self.sensors[question_type].query_count += 1

        self.query_cache[cache_key] = answer

        return answer, self.sensors[question_type].reliability

    def update_from_ground_truth(
        self,
        question_type: QuestionType,
        said_yes: bool,
        actual_truth: bool,
    ):
        """Update sensor reliability from observed ground truth."""
        self.sensors[question_type].update(said_yes, actual_truth)

    def get_posterior(
        self,
        question_type: QuestionType,
        prior: float,
        said_yes: bool,
    ) -> float:
        """Compute posterior probability given sensor reading."""
        return self.sensors[question_type].posterior(prior, said_yes)

    def clear_cache(self):
        """Clear query cache (call at start of each turn)."""
        self.query_cache = {}

    def _query_llm(self, question: str, context: str) -> bool:
        """Query LLM for yes/no answer."""
        prompt = f"""Answer this question about a text adventure game with only YES or NO.

Context:
{context}

Question: {question}

Answer (YES or NO):"""

        response = self.llm.complete(prompt).strip().upper()

        return response.startswith("YES")

    def get_all_stats(self) -> Dict[str, dict]:
        return {qt.value: sensor.get_stats() for qt, sensor in self.sensors.items()}


# =============================================================================
# BELIEF STATE
# =============================================================================

@dataclass
class BeliefState:
    """
    Agent's beliefs as probability distributions.

    Each belief is a probability that something is true.
    Updated via Bayes' rule from sensor readings and direct observations.
    """

    location_beliefs: Dict[str, float] = field(default_factory=dict)
    current_location: Optional[str] = None

    inventory_beliefs: Dict[str, float] = field(default_factory=dict)

    flag_beliefs: Dict[str, float] = field(default_factory=dict)

    goal_beliefs: Dict[str, float] = field(default_factory=dict)

    action_beliefs: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def update_belief(self, belief_name: str, category: str, new_probability: float):
        """Update a non-action belief to a new probability."""
        beliefs_map = {
            "location": self.location_beliefs,
            "inventory": self.inventory_beliefs,
            "flag": self.flag_beliefs,
            "goal": self.goal_beliefs,
        }
        target = beliefs_map.get(category)
        if target is not None:
            target[belief_name] = new_probability
            if category == "location" and self.location_beliefs:
                self.current_location = max(
                    self.location_beliefs, key=lambda k: self.location_beliefs[k]
                )

    def get_belief(self, belief_name: str, category: str, default: float = 0.5) -> float:
        """Get current belief probability (non-action categories)."""
        beliefs_map = {
            "location": self.location_beliefs,
            "inventory": self.inventory_beliefs,
            "flag": self.flag_beliefs,
            "goal": self.goal_beliefs,
        }
        target = beliefs_map.get(category)
        if target is not None:
            return target.get(belief_name, default)
        return default

    def set_certain(self, belief_name: str, category: str, value: bool):
        """Set a non-action belief to certainty (from direct observation)."""
        self.update_belief(belief_name, category, 1.0 if value else 0.0)

    def get_action_belief(self, action: str) -> Optional[Tuple[float, float]]:
        """Get Beta(alpha, beta) for an action, or None if not set."""
        return self.action_beliefs.get(action)

    def set_action_belief(self, action: str, alpha: float, beta: float):
        """Set Beta(alpha, beta) for an action."""
        self.action_beliefs[action] = (alpha, beta)

    def to_context_string(self) -> str:
        """Format beliefs for LLM context."""
        lines = []

        if self.current_location:
            conf = self.location_beliefs.get(self.current_location, 0)
            lines.append(f"Location: {self.current_location} (confidence: {conf:.0%})")

        has_items = [item for item, p in self.inventory_beliefs.items() if p > 0.7]
        if has_items:
            lines.append(f"Inventory: {', '.join(has_items)}")

        true_flags = [flag for flag, p in self.flag_beliefs.items() if p > 0.7]
        if true_flags:
            lines.append(f"Known state: {', '.join(true_flags)}")

        done_goals = [goal for goal, p in self.goal_beliefs.items() if p > 0.7]
        if done_goals:
            lines.append(f"Accomplished: {', '.join(done_goals)}")

        return "\n".join(lines) if lines else "No confident beliefs yet."


# =============================================================================
# DYNAMICS MODEL
# =============================================================================

@dataclass(frozen=True)
class StateActionKey:
    """Hashable key for state-action pairs."""
    state_hash: str
    action: str


@dataclass
class ObservedOutcome:
    """What we observed when taking an action."""
    next_state_hash: str
    reward: float
    observation_text: str = ""
    visit_count: int = 1
    reward_sum: float = 0.0

    def __post_init__(self):
        if self.reward_sum == 0.0 and self.visit_count == 1:
            self.reward_sum = self.reward

    @property
    def mean_reward(self) -> float:
        """Running mean reward for this (state, action) pair."""
        return self.reward_sum / self.visit_count if self.visit_count > 0 else 0.0


class DynamicsModel:
    """
    Learned dynamics from direct experience.

    In a deterministic game, one observation = certainty.
    State values computed via Bellman value iteration over the learned
    dynamics graph, cached with dirty flag.
    """

    def __init__(self, action_cost: float = 0.10, gamma: float = 0.95):
        """
        Args:
            action_cost: Cost per game action (same as UnifiedDecisionMaker).
            gamma: Discount factor. 0.95 models finite horizon (γ^100 ≈ 0.006).
                   Prevents divergence in cycles (go east / go west).
        """
        self.action_cost = action_cost
        self.gamma = gamma
        self.observations: Dict[StateActionKey, ObservedOutcome] = {}
        self.tried_actions: Set[str] = set()
        self.total_observations: int = 0
        self._state_n_total: Dict[str, int] = {}
        self._cached_values: Dict[str, float] = {}
        self._values_dirty: bool = True

    def has_observation(self, state_hash: str, action: str) -> bool:
        """Have we tried this action in this state?"""
        return StateActionKey(state_hash, action) in self.observations

    def get_observation(self, state_hash: str, action: str) -> Optional[ObservedOutcome]:
        """Get observed outcome if we have one."""
        return self.observations.get(StateActionKey(state_hash, action))

    def register_state(self, state_hash: str, n_total: int):
        """Register the total number of actions available in a state."""
        self._state_n_total[state_hash] = n_total
        self._values_dirty = True

    def record_observation(
        self,
        state_hash: str,
        action: str,
        next_state_hash: str,
        reward: float,
        observation_text: str = "",
    ):
        """Record an observed outcome, accumulating reward statistics."""
        key = StateActionKey(state_hash, action)
        existing = self.observations.get(key)
        if existing is not None:
            existing.visit_count += 1
            existing.reward_sum += reward
            existing.next_state_hash = next_state_hash
            existing.observation_text = observation_text
            existing.reward = existing.mean_reward
        else:
            self.observations[key] = ObservedOutcome(
                next_state_hash=next_state_hash,
                reward=reward,
                observation_text=observation_text,
            )
        self.tried_actions.add(action)
        self.total_observations += 1
        self._values_dirty = True

    def known_reward(self, state_hash: str, action: str) -> Optional[float]:
        """Get known mean reward for state-action pair. None if unobserved."""
        obs = self.get_observation(state_hash, action)
        return obs.mean_reward if obs else None

    def state_value(self, state_hash: str) -> float:
        """V(s): Bellman value of a state from learned dynamics.

        Returns 0.5 (max-entropy prior) for unknown states not in the
        dynamics graph.
        """
        if self._values_dirty:
            self._run_value_iteration()
        return self._cached_values.get(state_hash, 0.5)

    def _run_value_iteration(self):
        """Synchronous Bellman value iteration over the dynamics graph.

        V(s) = max_a Q(s, a)
        Q(s, a_tried)   = R(s,a) + γ·V(s') - c_act
        Q(s, a_untried)  = 1/N + γ·V_unknown - c_act
        V(s_unknown)     = 0.5   (Beta(1,1) max-entropy prior)
        """
        v_unknown = 0.5

        # Collect all states: registered states + successor states from observations
        all_states: Set[str] = set(self._state_n_total.keys())
        for key, obs in self.observations.items():
            all_states.add(key.state_hash)
            all_states.add(obs.next_state_hash)

        if not all_states:
            self._cached_values = {}
            self._values_dirty = False
            return

        # Build per-state tried actions index
        state_tried: Dict[str, List[StateActionKey]] = {}
        for key in self.observations:
            state_tried.setdefault(key.state_hash, []).append(key)

        # Initialize values
        values = {s: 0.0 for s in all_states}

        for _ in range(100):
            max_delta = 0.0
            new_values = {}
            for s in all_states:
                n_total = self._state_n_total.get(s, 0)
                tried_keys = state_tried.get(s, [])
                n_tried = len(tried_keys)

                # Compute Q-values for tried actions
                q_values = []
                for key in tried_keys:
                    obs = self.observations[key]
                    v_next = values.get(obs.next_state_hash, v_unknown)
                    q = obs.mean_reward + self.gamma * v_next - self.action_cost
                    q_values.append(q)

                # Q-value for untried actions (structural prior)
                n_untried = max(0, n_total - n_tried)
                if n_untried > 0:
                    q_untried = (1.0 / n_total) + self.gamma * v_unknown - self.action_cost
                    q_values.append(q_untried)

                if q_values:
                    new_v = max(q_values)
                else:
                    # State with no known actions and not registered
                    new_v = v_unknown

                new_values[s] = new_v
                max_delta = max(max_delta, abs(new_v - values[s]))

            values = new_values
            if max_delta < 1e-6:
                break

        self._cached_values = values
        self._values_dirty = False

    def get_stats(self) -> dict:
        return {
            "total_observations": self.total_observations,
            "unique_state_actions": len(self.observations),
            "unique_actions_tried": len(self.tried_actions),
        }


# =============================================================================
# UNIFIED DECISION MAKER
# =============================================================================

class UnifiedDecisionMaker:
    """
    Unified decision-making over game actions and LLM queries.

    Both action types evaluated by expected utility.
    """

    def __init__(self, question_cost: float = 0.01, action_cost: float = 0.10):
        """
        Args:
            question_cost: Cost of asking one LLM question (in reward units).
            action_cost: Cost of taking a game action (models limited turns).
        """
        self.question_cost = question_cost
        self.action_cost = action_cost
        self.v0 = 0.5  # V₀: value of unvisited state (Beta(1,1) prior)

    def choose(
        self,
        game_actions: List[str],
        possible_questions: List[Tuple[str, str]],  # (action, question)
        beliefs: Dict[str, Tuple[float, float]],     # action -> (alpha, beta)
        sensor: BinarySensor,
        dynamics: 'DynamicsModel',
        state_hash: str,
        categorical_sensor: Optional['CategoricalSensor'] = None,
        suggestion_cost: Optional[float] = None,
    ) -> Tuple[str, Any]:
        """
        Choose between asking a question or taking a game action.

        beliefs: Dict mapping action -> (alpha, beta) Beta parameters.
        Default prior for unknown actions: Beta(1/N, 1-1/N) where N = len(game_actions).

        Returns: ('ask', (action, question)) or ('take', action) or ('suggest', None)
        """
        n = len(game_actions)
        default_prior = (1.0 / n, 1.0 - 1.0 / n)

        # Compute EU for all game actions (Q-values with state value)
        game_eus = {}
        for action in game_actions:
            obs = dynamics.get_observation(state_hash, action)
            if obs is not None:
                v_next = dynamics.state_value(obs.next_state_hash)
                game_eus[action] = obs.reward + v_next - self.action_cost
            else:
                alpha, beta = beliefs.get(action, default_prior)
                game_eus[action] = alpha / (alpha + beta) + self.v0 - self.action_cost

        best_game_action = max(game_actions, key=lambda a: game_eus[a])

        # Compute EU for all binary questions (VOI - cost)
        best_question = None
        best_question_eu = float('-inf')

        for action, question in possible_questions:
            if dynamics.has_observation(state_hash, action):
                continue

            alpha, beta = beliefs.get(action, default_prior)
            voi = self.compute_voi(action, alpha, beta, sensor, game_eus)
            question_eu = voi - self.question_cost

            if question_eu > best_question_eu:
                best_question_eu = question_eu
                best_question = (action, question)

        # Compute EU for categorical suggestion (VOI - cost)
        cat_eu = float('-inf')
        if categorical_sensor is not None:
            s_cost = suggestion_cost if suggestion_cost is not None else self.question_cost
            cat_voi = self.compute_voi_categorical(game_actions, beliefs, categorical_sensor)
            cat_eu = cat_voi - s_cost

        # Pick best among: take, ask (binary), suggest (categorical)
        candidates = [('take', best_game_action, game_eus[best_game_action])]
        if best_question is not None and best_question_eu > 0:
            candidates.append(('ask', best_question, game_eus[best_game_action] + best_question_eu))
        if cat_eu > 0:
            candidates.append(('suggest', None, game_eus[best_game_action] + cat_eu))

        best = max(candidates, key=lambda c: c[2])
        return (best[0], best[1])

    def compute_voi(
        self,
        action: str,
        alpha: float,
        beta: float,
        sensor: BinarySensor,
        all_game_eus: Dict[str, float],
    ) -> float:
        """
        Compute value of information for asking about this action.

        Uses Beta parameters (not just the mean) — two actions with the
        same mean but different total counts have different VOI because
        the uncertain one has more to learn from a question.

        VOI = E[max EU after asking] - max EU now
        """
        current_belief = alpha / (alpha + beta)
        current_best_eu = max(all_game_eus.values())

        # P(LLM says yes)
        p_yes = sensor.tpr * current_belief + sensor.fpr * (1 - current_belief)
        p_no = 1 - p_yes

        # Posterior beliefs after each answer
        posterior_if_yes = sensor.posterior(current_belief, said_yes=True)
        posterior_if_no = sensor.posterior(current_belief, said_yes=False)

        # EU of this action under each posterior (includes V₀ and action cost)
        eu_if_yes = posterior_if_yes * 1.0 + self.v0 - self.action_cost
        eu_if_no = posterior_if_no * 1.0 + self.v0 - self.action_cost

        # Best EU achievable after each answer
        other_best = max(
            (eu for a, eu in all_game_eus.items() if a != action),
            default=0.0
        )

        best_eu_if_yes = max(eu_if_yes, other_best)
        best_eu_if_no = max(eu_if_no, other_best)

        # Expected best EU after asking
        expected_best_after = p_yes * best_eu_if_yes + p_no * best_eu_if_no

        # VOI = improvement over current best
        voi = expected_best_after - current_best_eu

        return max(0.0, voi)

    def compute_voi_categorical(
        self,
        actions: List[str],
        beliefs: Dict[str, Tuple[float, float]],
        sensor: 'CategoricalSensor',
    ) -> float:
        """
        VOI for 'what should I do?' — one question updates all N beliefs.

        For each possible suggestion, compute: P(LLM suggests it) and the
        posterior belief over all actions. Then compute expected best EU
        after observing the suggestion.
        """
        n = len(actions)
        acc = sensor.accuracy
        wrong_prob = (1.0 - acc) / max(n - 1, 1)
        default_prior = (1.0 / n, 1.0 - 1.0 / n)

        priors = {
            a: beliefs.get(a, default_prior)[0] / sum(beliefs.get(a, default_prior))
            for a in actions
        }
        current_best_eu = max(p - self.action_cost for p in priors.values())

        expected_best_after = 0.0
        for suggested in actions:
            # P(LLM suggests this action) = sum over true actions of P(suggest|true)*P(true)
            p_suggests = sum(
                (acc if a == suggested else wrong_prob) * priors[a]
                for a in actions
            )
            post = sensor.posteriors(priors, suggested)
            best_eu_after = max(post[a] - self.action_cost for a in actions)
            expected_best_after += p_suggests * best_eu_after

        return max(0.0, expected_best_after - current_best_eu)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Bayesian IF Agent v6 - Core Unit Tests")
    print("=" * 60)

    # Test 1: BinarySensor (SPEC v6: Beta(2,1) / Beta(1,2))
    sensor = BinarySensor()
    assert abs(sensor.tpr - 2/3) < 0.01
    assert abs(sensor.fpr - 1/3) < 0.01
    assert abs(sensor.reliability - 1/3) < 0.01
    print(f"Sensor: TPR={sensor.tpr:.2f}, FPR={sensor.fpr:.2f}, rel={sensor.reliability:.2f}")

    # Update with true positive
    sensor.update(said_yes=True, was_true=True)
    assert sensor.tpr > 2/3
    assert sensor.ground_truth_count == 1
    print(f"After TP: TPR={sensor.tpr:.2f}, FPR={sensor.fpr:.2f}")

    # Posterior math
    posterior = sensor.posterior(0.5, said_yes=True)
    assert posterior > 0.5  # Saying yes should increase belief
    posterior_no = sensor.posterior(0.5, said_yes=False)
    assert posterior_no < 0.5  # Saying no should decrease belief
    print(f"Posterior(0.5, yes)={posterior:.3f}, Posterior(0.5, no)={posterior_no:.3f}")

    # Test 2: QuestionType
    assert len(QuestionType) == 9
    assert QuestionType.ACTION_HELPS.value == "action_helps"
    assert QuestionType.SUGGEST_ACTION.value == "suggest_action"
    assert QuestionType.MADE_PROGRESS.value == "made_progress"
    print(f"\nQuestionTypes: {[qt.value for qt in QuestionType]}")

    # Test 3: LLMSensorBank with mock (all sensors use spec defaults)
    class MockLLM:
        def complete(self, prompt):
            return "YES"

    bank = LLMSensorBank(MockLLM())
    answer, rel = bank.ask(QuestionType.ACTION_HELPS, "Will 'go north' help?", "In a room.")
    assert answer is True
    assert isinstance(rel, float)
    print(f"\nSensorBank ask: answer={answer}, reliability={rel:.2f}")

    # Cache hit
    answer2, _ = bank.ask(QuestionType.ACTION_HELPS, "Will 'go north' help?", "In a room.")
    assert answer2 == answer
    bank.clear_cache()
    print("Cache works correctly")

    # Ground truth update
    bank.update_from_ground_truth(QuestionType.ACTION_HELPS, said_yes=True, actual_truth=True)
    stats = bank.get_all_stats()
    assert stats["action_helps"]["ground_truths"] == 1
    print(f"Sensor stats: {stats['action_helps']}")

    # Test 4: BeliefState
    beliefs = BeliefState()
    assert beliefs.to_context_string() == "No confident beliefs yet."

    beliefs.update_belief("bedroom", "location", 0.9)
    assert beliefs.current_location == "bedroom"
    assert beliefs.get_belief("bedroom", "location") == 0.9
    assert beliefs.get_belief("kitchen", "location") == 0.5  # default

    beliefs.set_certain("keys", "inventory", True)
    assert beliefs.get_belief("keys", "inventory") == 1.0

    beliefs.set_certain("wallet", "inventory", False)
    assert beliefs.get_belief("wallet", "inventory") == 0.0

    # Action beliefs are Beta tuples
    beliefs.set_action_belief("go north", 2.0, 8.0)
    ab = beliefs.get_action_belief("go north")
    assert ab == (2.0, 8.0)
    assert beliefs.get_action_belief("nonexistent") is None

    ctx = beliefs.to_context_string()
    assert "bedroom" in ctx
    assert "keys" in ctx
    print(f"\nBeliefs context:\n{ctx}")

    # Test 5: StateActionKey
    key1 = StateActionKey("hash1", "go north")
    key2 = StateActionKey("hash1", "go north")
    assert key1 == key2
    assert hash(key1) == hash(key2)
    d = {key1: "value"}
    assert d[key2] == "value"
    print(f"\nStateActionKey: frozen and hashable")

    # Test 6: DynamicsModel
    dynamics = DynamicsModel()
    assert not dynamics.has_observation("s1", "go")
    assert dynamics.known_reward("s1", "go") is None

    dynamics.record_observation("s1", "go", "s2", 1.0, "You go north.")
    assert dynamics.has_observation("s1", "go")
    assert dynamics.known_reward("s1", "go") == 1.0
    assert "go" in dynamics.tried_actions

    # Accumulate (running mean reward)
    dynamics.record_observation("s1", "go", "s2", 1.0, "You go north again.")
    assert dynamics.total_observations == 2
    stats = dynamics.get_stats()
    assert stats["unique_state_actions"] == 1
    obs = dynamics.get_observation("s1", "go")
    assert obs.visit_count == 2
    assert obs.mean_reward == 1.0  # both visits had r=1.0
    print(f"\nDynamics stats: {stats}")

    # Test 7: UnifiedDecisionMaker (no action_prior — uses 1/N from beliefs)
    dm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)

    # When all actions known, should take best known
    dynamics2 = DynamicsModel()
    dynamics2.record_observation("s", "a1", "s2", 5.0)
    dynamics2.record_observation("s", "a2", "s", 0.0)
    dynamics2.register_state("s", 2)
    dynamics2.register_state("s2", 1)  # register successor

    decision = dm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=BinarySensor(),
        dynamics=dynamics2,
        state_hash="s",
    )
    assert decision == ('take', 'a1')
    print(f"\nDecision (known): {decision}")

    # When actions unknown with reliable sensor, asking can win
    dynamics3 = DynamicsModel()
    sensor3 = BinarySensor(tp_alpha=9, tp_beta=1, fp_alpha=1, fp_beta=9)

    decision2 = dm.choose(
        game_actions=["a1", "a2"],
        possible_questions=[("a1", "Will a1 help?"), ("a2", "Will a2 help?")],
        beliefs={"a1": (0.5, 0.5), "a2": (0.5, 0.5)},
        sensor=sensor3,
        dynamics=dynamics3,
        state_hash="s",
    )
    print(f"Decision (unknown, reliable sensor): {decision2}")

    # VOI computation (now takes alpha, beta instead of point estimate)
    # EUs include V₀: belief_mean + V₀ - action_cost
    voi = dm.compute_voi("a1", 0.5, 0.5, sensor3, {"a1": 0.5 + 0.5 - 0.10, "a2": 0.5 + 0.5 - 0.10})
    assert voi >= 0
    print(f"VOI: {voi:.4f}")

    # Test 8: CategoricalSensor (Beta(1,1) max-entropy prior)
    cat = CategoricalSensor()
    assert abs(cat.accuracy - 0.5) < 0.01
    assert cat.query_count == 0
    assert cat.ground_truth_count == 0
    print(f"\nCategoricalSensor: accuracy={cat.accuracy:.2f}")

    # Posteriors sum to 1
    priors = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
    post = cat.posteriors(priors, "a")
    assert abs(sum(post.values()) - 1.0) < 1e-9
    print(f"Posteriors sum: {sum(post.values()):.6f}")

    # Suggested action gets boosted
    assert post["a"] > priors["a"]
    # Non-suggested actions drop
    assert post["b"] < priors["b"]
    assert post["c"] < priors["c"]
    print(f"Posterior(a)={post['a']:.3f} > prior={priors['a']:.3f}")

    # Accuracy updates from ground truth
    cat.update(suggested_correct=True)
    assert cat.accuracy > 0.5
    cat.update(suggested_correct=False)
    cat.update(suggested_correct=False)
    # After 1 correct, 2 incorrect from prior Beta(1,1): Beta(2,3), mean 0.4
    assert abs(cat.accuracy - 0.4) < 0.01
    assert cat.ground_truth_count == 3
    print(f"After 3 updates: accuracy={cat.accuracy:.2f}")

    # get_stats
    stats = cat.get_stats()
    assert "accuracy" in stats
    assert "queries" in stats
    assert "ground_truths" in stats
    print(f"CategoricalSensor stats: {stats}")

    # Test 9: VOI categorical
    dm2 = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)
    cat2 = CategoricalSensor(accuracy_alpha=8, accuracy_beta=2)  # accuracy 0.8
    voi_cat = dm2.compute_voi_categorical(
        ["a1", "a2", "a3"],
        {"a1": (0.33, 0.67), "a2": (0.33, 0.67), "a3": (0.33, 0.67)},
        cat2,
    )
    assert voi_cat >= 0
    print(f"\nVOI categorical: {voi_cat:.4f}")

    # VOI should be > 0 with weak uniform priors
    assert voi_cat > 0
    print("VOI categorical > 0 with weak priors: OK")

    print("\nAll core v5 tests passed!")
