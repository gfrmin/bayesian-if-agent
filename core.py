"""
Bayesian Interactive Fiction Agent

An agent that maintains beliefs about game dynamics and selects actions
to maximise expected utility (points/survival).

Key design principles:
- Jericho provides GROUND TRUTH state (location, inventory, world hash)
- LLM provides SENSOR readings (action relevance scores)
- Agent LEARNS the LLM's reliability from experience
- Bayesian updating combines sensor prior with dynamics posterior
- Expected utility maximisation selects ACTIONS

What's FIXED:
- Bayesian inference as update rule
- Expected utility as action selection
- Game interface: state from Jericho, score signal

What's LEARNED:
- Dynamics: P(outcome | state, action)
- LLM reliability: P(LLM_score | action_valuable)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random
import math


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

@dataclass(frozen=True)
class GameState:
    """
    Game state from Jericho ground truth.

    location: int (Jericho location ID)
    inventory: frozenset of item names
    world_hash: str (Jericho state hash for full disambiguation)
    """
    location: int
    inventory: frozenset
    world_hash: str

    def __str__(self):
        return f"State(loc={self.location}, inv={len(self.inventory)} items)"

    @staticmethod
    def initial():
        return GameState(
            location=0,
            inventory=frozenset(),
            world_hash=""
        )


@dataclass(frozen=True)
class Outcome:
    """Result of taking an action."""
    next_state: GameState
    reward: float


@dataclass(frozen=True)
class Transition:
    """A single observed transition with raw observation for history."""
    state: GameState
    action: str
    next_state: GameState
    reward: float
    raw_observation: str


# =============================================================================
# DYNAMICS MODEL
# =============================================================================

class DynamicsModel:
    """
    Learns P(next_state, reward | current_state, action).

    Uses a simple count-based model with pseudocounts for smoothing.
    """

    def __init__(self, prior_pseudocount: float = 0.1):
        self.prior_pseudocount = prior_pseudocount

        # Counts: (state, action) -> {(next_state, reward): count}
        self.transition_counts: Dict[Tuple[GameState, str], Dict[Tuple[GameState, float], float]] = \
            defaultdict(lambda: defaultdict(lambda: self.prior_pseudocount))

        # Total counts per (state, action) for normalisation
        self.total_counts: Dict[Tuple[GameState, str], float] = \
            defaultdict(lambda: self.prior_pseudocount)

        # Track what we've actually observed (vs just prior)
        self.observed_transitions: Set[Tuple[GameState, str, GameState, float]] = set()

        # Full transition history for contradiction detection
        self.history: List[Transition] = []

    def update(self, state: GameState, action: str, next_state: GameState,
               reward: float, raw_observation: str = ""):
        """Record an observed transition."""
        key = (state, action)
        outcome = (next_state, reward)

        self.transition_counts[key][outcome] += 1.0
        self.total_counts[key] += 1.0
        self.observed_transitions.add((state, action, next_state, reward))

        self.history.append(Transition(
            state=state, action=action, next_state=next_state,
            reward=reward, raw_observation=raw_observation
        ))

    def predict(self, state: GameState, action: str) -> Dict[Tuple[GameState, float], float]:
        """
        P(next_state, reward | state, action)

        Returns distribution over (next_state, reward) pairs.
        """
        key = (state, action)
        total = self.total_counts[key]

        if total <= self.prior_pseudocount:
            return {}

        return {
            outcome: count / total
            for outcome, count in self.transition_counts[key].items()
        }

    def sample_outcome(self, state: GameState, action: str) -> Optional[Tuple[GameState, float]]:
        """Sample from P(outcome | state, action). For Thompson sampling."""
        dist = self.predict(state, action)
        if not dist:
            return None

        outcomes, probs = zip(*dist.items())
        return random.choices(outcomes, probs)[0]

    def expected_reward(self, state: GameState, action: str) -> float:
        """E[reward | state, action]"""
        dist = self.predict(state, action)
        if not dist:
            return 0.0
        return sum(prob * reward for (_, reward), prob in dist.items())

    def uncertainty(self, state: GameState, action: str) -> float:
        """
        How uncertain are we about this transition?

        Returns entropy of the distribution, or high value if no observations.
        """
        dist = self.predict(state, action)
        if not dist:
            return 10.0
        return -sum(p * math.log(p + 1e-10) for p in dist.values())

    def observation_count(self, state: GameState, action: str) -> int:
        """How many times have we seen this (state, action) pair?"""
        key = (state, action)
        total = self.total_counts[key]
        return max(0, int(total - self.prior_pseudocount))

    def get_outcome_counts(self, state: GameState, action: str) -> Dict[Tuple[GameState, float], int]:
        """Return actual observation counts (excluding pseudocounts) for each outcome."""
        key = (state, action)
        return {
            outcome: max(0, int(count - self.prior_pseudocount))
            for outcome, count in self.transition_counts[key].items()
            if count > self.prior_pseudocount
        }


# =============================================================================
# ACTION SELECTION
# =============================================================================

class BayesianActionSelector:
    """
    Selects actions using:
    - Prior from LLM sensor (treated as noisy observation)
    - Learned dynamics (from experience)
    - Learned LLM reliability (via sensor_model)

    sensor and sensor_model are injected — this class remains stdlib-only.
    When sensor is None, behaves as if all actions have uniform 0.5 relevance.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        exploration_weight: float = 0.1,
        sensor=None,
        sensor_model=None,
        base_prior: float = 0.1,
    ):
        self.dynamics = dynamics
        self.exploration_weight = exploration_weight
        self.sensor = sensor          # object with get_relevance_scores()
        self.sensor_model = sensor_model  # LLMSensorModel instance
        self.base_prior = base_prior

        # Cache LLM scores for current turn
        self.current_llm_scores: Dict[str, float] = {}

    def select_action(
        self,
        state: GameState,
        valid_actions: List[str],
        observation: str,
    ) -> str:
        """
        Select action to maximise expected utility.

        Uses Thompson sampling for exploration.
        """
        if not valid_actions:
            return "look"

        # Get LLM relevance scores
        if self.sensor is not None:
            self.current_llm_scores = self.sensor.get_relevance_scores(
                observation, valid_actions
            )
        else:
            self.current_llm_scores = {a: 0.5 for a in valid_actions}

        action_values = {}

        for action in valid_actions:
            llm_score = self.current_llm_scores.get(action, 0.5)
            n_obs = self.dynamics.observation_count(state, action)

            if n_obs > 0:
                # We have experience: blend learned value with fading prior
                learned_value = self.dynamics.expected_reward(state, action)
                prior_value = self._llm_score_to_value(llm_score)

                # Prior weight decays with experience
                prior_weight = 1.0 / (1.0 + n_obs)
                experience_weight = 1.0 - prior_weight

                value = experience_weight * learned_value + prior_weight * prior_value

                # Add exploration bonus for uncertainty
                uncertainty = self.dynamics.uncertainty(state, action)
                value += self.exploration_weight * uncertainty
            else:
                # No experience: use LLM-informed prior
                value = self._llm_score_to_value(llm_score)

                # High exploration bonus for untried actions
                value += self.exploration_weight * 2.0

            action_values[action] = value

        # Thompson sampling: add noise, pick best
        return self._thompson_sample(action_values)

    def _llm_score_to_value(self, llm_score: float) -> float:
        """
        Convert LLM score to expected value using sensor model.

        P(valuable | LLM_score) via Bayes, then expected value.
        When no sensor_model, use the raw score scaled down.
        """
        if self.sensor_model is not None:
            p_valuable = self.sensor_model.posterior_valuable(
                llm_score, self.base_prior
            )
        else:
            # Without sensor model, scale LLM score directly
            p_valuable = llm_score * self.base_prior * 2

        # Expected value: P(valuable) * value_if_valuable
        return p_valuable * 1.0

    def _thompson_sample(self, action_values: Dict[str, float]) -> str:
        """Thompson sampling: add Gaussian noise, pick best."""
        noisy_values = {
            action: value + random.gauss(0, 0.1)
            for action, value in action_values.items()
        }
        return max(noisy_values, key=noisy_values.get)

    def observe_outcome(self, state: GameState, action: str, outcome: Outcome,
                        raw_observation: str = ""):
        """
        Learn from outcome.
        Update both dynamics AND LLM sensor model.
        """
        self.dynamics.update(state, action, outcome.next_state,
                             outcome.reward, raw_observation)

        # Update LLM sensor model
        if self.sensor_model is not None:
            llm_score = self.current_llm_scores.get(action, 0.5)
            self.sensor_model.update(llm_score, outcome.reward)

    def get_llm_score(self, action: str) -> float:
        """Get cached LLM score for action."""
        return self.current_llm_scores.get(action, 0.5)


# =============================================================================
# THE AGENT
# =============================================================================

class BayesianIFAgent:
    """
    Bayesian agent for interactive fiction.

    Uses LLM as sensor for action relevance.
    Learns dynamics and LLM reliability from experience.
    State is set externally (from Jericho ground truth).
    """

    def __init__(
        self,
        sensor=None,
        sensor_model=None,
        pseudocount: float = 0.1,
        exploration_weight: float = 0.1,
    ):
        self.dynamics = DynamicsModel(prior_pseudocount=pseudocount)
        self.selector = BayesianActionSelector(
            dynamics=self.dynamics,
            exploration_weight=exploration_weight,
            sensor=sensor,
            sensor_model=sensor_model,
        )
        self.sensor_model = sensor_model

        # State tracking
        self.current_state: Optional[GameState] = None
        self.previous_score: float = 0.0
        self.history: List[Dict] = []

    def observe(self, state: GameState, observation: str, score: float):
        """
        Process a new observation from the game.

        Takes a GameState directly (from Jericho ground truth).
        """
        reward = score - self.previous_score

        # If we have a previous state and action, update dynamics
        if self.current_state is not None and self.history:
            last_action = self.history[-1].get("action")
            if last_action:
                outcome = Outcome(next_state=state, reward=reward)
                self.selector.observe_outcome(
                    self.current_state, last_action, outcome,
                    raw_observation=observation
                )

        self.current_state = state
        self.previous_score = score

        self.history.append({
            "observation": observation[:200],
            "state": str(state),
            "score": score,
            "reward": reward,
        })

    def act(self, candidate_actions: List[str], observation: str = "",
            budget=None) -> str:
        """
        Select an action to take.

        If budget (a metareason.ComputationBudget) is provided, uses the
        metareasoner's deliberate() instead of a single selection.

        Returns the chosen action.
        """
        if self.current_state is None:
            return random.choice(candidate_actions) if candidate_actions else "look"

        if budget is not None:
            from metareason import deliberate
            action, _meta = deliberate(
                self.current_state, self.dynamics, self.selector,
                candidate_actions, budget, observation
            )
        else:
            action = self.selector.select_action(
                self.current_state, candidate_actions, observation
            )

        if self.history:
            self.history[-1]["action"] = action

        return action

    def get_statistics(self) -> Dict:
        """Return statistics about the agent's learning."""
        stats = {
            "total_steps": len(self.history),
            "unique_states_visited": len(set(h.get("state") for h in self.history)),
            "transitions_learned": len(self.dynamics.observed_transitions),
            "dynamics_history_size": len(self.dynamics.history),
            "current_state": str(self.current_state),
            "current_score": self.previous_score,
        }
        if self.sensor_model is not None:
            stats["sensor_stats"] = self.sensor_model.get_statistics()
        return stats


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Bayesian IF Agent - Core Unit Tests")
    print("=" * 60)

    # Test 1: GameState
    state1 = GameState(location=5, inventory=frozenset(["keys"]), world_hash="abc123")
    state2 = GameState(location=5, inventory=frozenset(["keys", "wallet"]), world_hash="def456")
    print(f"State 1: {state1}")
    print(f"State 2: {state2}")
    assert state1 != state2
    assert hash(state1) != hash(state2)

    # Test GameState.initial()
    initial = GameState.initial()
    assert initial.location == 0
    assert initial.inventory == frozenset()
    print(f"Initial: {initial}")

    # Test 2: Outcome
    outcome = Outcome(next_state=state2, reward=5.0)
    print(f"\nOutcome: {outcome}")

    # Test 3: DynamicsModel
    dynamics = DynamicsModel()
    dynamics.update(state1, "take wallet", state2, 0.0)
    dynamics.update(state1, "take wallet", state2, 0.0)

    pred = dynamics.predict(state1, "take wallet")
    assert len(pred) > 0
    print(f"\nDynamics prediction for 'take wallet': {len(pred)} outcomes")
    print(f"Uncertainty: {dynamics.uncertainty(state1, 'take wallet'):.3f}")
    print(f"Observation count: {dynamics.observation_count(state1, 'take wallet')}")
    print(f"Expected reward: {dynamics.expected_reward(state1, 'take wallet'):.3f}")

    # Test 4: BayesianActionSelector without sensor (uniform prior)
    selector = BayesianActionSelector(dynamics=dynamics, exploration_weight=0.2)
    action = selector.select_action(state1, ["take wallet", "go north", "look"], "You are in a room.")
    print(f"\nSelected action (no sensor): {action}")
    assert action in ["take wallet", "go north", "look"]

    # Test 5: BayesianActionSelector with mock sensor
    class MockSensor:
        def get_relevance_scores(self, obs, actions):
            return {a: 0.9 if "wallet" in a else 0.1 for a in actions}

    from sensor_model import LLMSensorModel
    sm = LLMSensorModel()
    selector2 = BayesianActionSelector(
        dynamics=dynamics, exploration_weight=0.1,
        sensor=MockSensor(), sensor_model=sm,
    )
    # Run multiple times to check bias
    counts = defaultdict(int)
    for _ in range(100):
        a = selector2.select_action(state1, ["take wallet", "go north", "look"], "Room.")
        counts[a] += 1
    print(f"Action selection distribution (100 trials): {dict(counts)}")

    # Test 6: observe_outcome updates dynamics + sensor
    selector2.observe_outcome(
        state1, "take wallet",
        Outcome(next_state=state2, reward=1.0),
        raw_observation="Taken."
    )
    assert dynamics.observation_count(state1, "take wallet") == 3  # 2 earlier + 1 now

    # Test 7: BayesianIFAgent
    agent = BayesianIFAgent(pseudocount=0.1, exploration_weight=0.2)
    s0 = GameState(location=1, inventory=frozenset(), world_hash="h0")
    agent.observe(s0, "You wake up.", 0.0)
    action = agent.act(["stand up", "sleep", "look"], "You wake up.")
    print(f"\nAgent chose: {action}")

    s1 = GameState(location=1, inventory=frozenset(), world_hash="h1")
    agent.observe(s1, "You stand up.", 1.0)
    stats = agent.get_statistics()
    print(f"Agent stats: {stats}")

    print("\nAll core tests passed!")
