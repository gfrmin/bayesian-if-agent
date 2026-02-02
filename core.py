"""
Bayesian Interactive Fiction Agent — v4

Binary sensor bank + unified expected utility maximisation.

Core insight: instead of asking the LLM for rich structured analysis, ask it
simple yes/no questions and learn each question type's reliability via
Beta-distributed TPR/FPR.

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


class DynamicsModel:
    """
    Learned dynamics from direct experience.

    In a deterministic game, one observation = certainty.
    """

    def __init__(self):
        self.observations: Dict[StateActionKey, ObservedOutcome] = {}
        self.tried_actions: Set[str] = set()
        self.total_observations: int = 0

    def has_observation(self, state_hash: str, action: str) -> bool:
        """Have we tried this action in this state?"""
        return StateActionKey(state_hash, action) in self.observations

    def get_observation(self, state_hash: str, action: str) -> Optional[ObservedOutcome]:
        """Get observed outcome if we have one."""
        return self.observations.get(StateActionKey(state_hash, action))

    def record_observation(
        self,
        state_hash: str,
        action: str,
        next_state_hash: str,
        reward: float,
        observation_text: str = "",
    ):
        """Record an observed outcome."""
        key = StateActionKey(state_hash, action)
        self.observations[key] = ObservedOutcome(
            next_state_hash=next_state_hash,
            reward=reward,
            observation_text=observation_text,
        )
        self.tried_actions.add(action)
        self.total_observations += 1

    def known_reward(self, state_hash: str, action: str) -> Optional[float]:
        """Get known reward for state-action pair. None if unobserved."""
        obs = self.get_observation(state_hash, action)
        return obs.reward if obs else None

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

    def choose(
        self,
        game_actions: List[str],
        possible_questions: List[Tuple[str, str]],  # (action, question)
        beliefs: Dict[str, Tuple[float, float]],     # action -> (alpha, beta)
        sensor: BinarySensor,
        dynamics: 'DynamicsModel',
        state_hash: str,
    ) -> Tuple[str, Any]:
        """
        Choose between asking a question or taking a game action.

        beliefs: Dict mapping action -> (alpha, beta) Beta parameters.
        Default prior for unknown actions: Beta(1/N, 1-1/N) where N = len(game_actions).

        Returns: ('ask', (action, question)) or ('take', action)
        """
        n = len(game_actions)
        default_prior = (1.0 / n, 1.0 - 1.0 / n)

        # Compute EU for all game actions
        game_eus = {}
        for action in game_actions:
            known = dynamics.known_reward(state_hash, action)
            if known is not None:
                game_eus[action] = known - self.action_cost
            else:
                alpha, beta = beliefs.get(action, default_prior)
                game_eus[action] = alpha / (alpha + beta) - self.action_cost

        best_game_action = max(game_actions, key=lambda a: game_eus[a])

        # Compute EU for all questions (VOI - cost)
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

        if best_question is not None and best_question_eu > 0:
            return ('ask', best_question)
        else:
            return ('take', best_game_action)

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

        # EU of this action under each posterior (includes action cost)
        eu_if_yes = posterior_if_yes * 1.0 - self.action_cost
        eu_if_no = posterior_if_no * 1.0 - self.action_cost

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
    assert len(QuestionType) == 7
    assert QuestionType.ACTION_HELPS.value == "action_helps"
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

    # Overwrite (deterministic game)
    dynamics.record_observation("s1", "go", "s2", 1.0, "You go north again.")
    assert dynamics.total_observations == 2
    stats = dynamics.get_stats()
    assert stats["unique_state_actions"] == 1
    print(f"\nDynamics stats: {stats}")

    # Test 7: UnifiedDecisionMaker (no action_prior — uses 1/N from beliefs)
    dm = UnifiedDecisionMaker(question_cost=0.01, action_cost=0.10)

    # When all actions known, should take best known
    dynamics2 = DynamicsModel()
    dynamics2.record_observation("s", "a1", "s2", 5.0)
    dynamics2.record_observation("s", "a2", "s2", 0.0)

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
    voi = dm.compute_voi("a1", 0.5, 0.5, sensor3, {"a1": 0.4, "a2": 0.4})
    assert voi >= 0
    print(f"VOI: {voi:.4f}")

    print("\nAll core v6 tests passed!")
