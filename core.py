"""
Bayesian Interactive Fiction Agent — v3

An agent that maintains rich beliefs about the game situation and uses
an LLM oracle for structured understanding.

Key design principles:
- Jericho provides GROUND TRUTH state (location, inventory, world hash)
- LLM provides ORACLE analysis (goals, blockers, recommendations)
- Agent LEARNS the oracle's reliability from experience
- Bayesian updating combines oracle observations with dynamics posterior
- Expected utility maximisation selects ACTIONS

What's FIXED:
- Bayesian inference as update rule
- Expected utility as action selection
- Game interface: state from Jericho, score signal

What's LEARNED:
- Dynamics: P(outcome | state, action)
- Oracle reliability: per-query-type accuracy
- Beliefs: goals, blockers, state variables
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
import random
import math


# =============================================================================
# STATE REPRESENTATION (unchanged from v2)
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
# AGENT BELIEFS (v3)
# =============================================================================

@dataclass
class AgentBeliefs:
    """
    Everything the agent believes about the current situation.

    This gets passed to the LLM for context.
    This gets updated based on LLM responses.
    """

    # Current state
    location: str = "unknown"
    inventory: List[str] = field(default_factory=list)

    # Goal tracking
    overall_goal: str = "unknown"
    current_subgoal: str = "unknown"
    accomplished: List[str] = field(default_factory=list)

    # Blocker tracking
    blocking_condition: Optional[str] = None
    failed_actions: Dict[str, str] = field(default_factory=dict)  # action -> reason

    # State variables we're tracking
    tracked_state: Dict[str, Any] = field(default_factory=dict)

    # History
    action_history: List[str] = field(default_factory=list)
    observation_history: List[str] = field(default_factory=list)

    # Internal (for state key derivation)
    _location_id: int = 0
    _world_hash: str = ""

    def to_prompt_context(self) -> str:
        """Format beliefs for inclusion in LLM prompt."""
        sections = []

        sections.append(f"LOCATION: {self.location}")
        sections.append(f"INVENTORY: {', '.join(self.inventory) if self.inventory else 'empty'}")
        sections.append(f"OVERALL GOAL: {self.overall_goal}")
        sections.append(f"CURRENT SUBGOAL: {self.current_subgoal}")

        if self.accomplished:
            sections.append(f"ACCOMPLISHED: {', '.join(self.accomplished)}")

        if self.blocking_condition:
            sections.append(f"BLOCKED BY: {self.blocking_condition}")

        if self.failed_actions:
            failed = [f"'{a}' ({r})" for a, r in list(self.failed_actions.items())[-5:]]
            sections.append(f"RECENT FAILURES: {'; '.join(failed)}")

        if self.tracked_state:
            state_str = ', '.join(f"{k}={v}" for k, v in self.tracked_state.items())
            sections.append(f"TRACKED STATE: {state_str}")

        if self.action_history:
            recent = self.action_history[-10:]
            sections.append(f"RECENT ACTIONS: {' -> '.join(recent)}")

        return '\n'.join(sections)

    def to_state_key(self) -> GameState:
        """Derive a frozen hashable key from current beliefs."""
        return GameState(
            location=self._location_id,
            inventory=frozenset(self.inventory),
            world_hash=self._world_hash,
        )


@dataclass
class SituationUnderstanding:
    """
    LLM's analysis of the current situation.

    This is an OBSERVATION — data to condition on.
    """

    # Goal understanding
    overall_goal: Optional[str] = None
    immediate_goal: Optional[str] = None

    # Progress understanding
    accomplished: List[str] = field(default_factory=list)
    progress_made: bool = False

    # Blocker understanding
    blocking_condition: Optional[str] = None
    blocking_reason: Optional[str] = None

    # Action recommendation
    recommended_action: Optional[str] = None
    action_reasoning: Optional[str] = None
    alternative_actions: List[str] = field(default_factory=list)

    # State suggestions
    important_state_variables: List[str] = field(default_factory=list)

    # Confidence (self-reported by LLM, take with grain of salt)
    confidence: float = 0.5

    @classmethod
    def from_json(cls, data: Dict) -> 'SituationUnderstanding':
        """Parse from LLM JSON response."""
        return cls(
            overall_goal=data.get("overall_goal"),
            immediate_goal=data.get("immediate_goal"),
            accomplished=data.get("accomplished", []),
            progress_made=data.get("progress_made", False),
            blocking_condition=data.get("blocking_condition"),
            blocking_reason=data.get("blocking_reason"),
            recommended_action=data.get("recommended_action"),
            action_reasoning=data.get("action_reasoning"),
            alternative_actions=data.get("alternative_actions", []),
            important_state_variables=data.get("important_state_variables", []),
            confidence=data.get("confidence", 0.5),
        )


# =============================================================================
# ORACLE RELIABILITY TRACKING (v3)
# =============================================================================

@dataclass
class OracleReliability:
    """
    Track how reliable different types of LLM responses are.

    We learn this from experience.
    """

    # Recommendation accuracy: when LLM recommends action, does it help?
    recommendations_followed: int = 0
    recommendations_helped: int = 0

    # Goal inference accuracy: does pursuing inferred goals lead to score?
    goals_pursued: int = 0
    goals_achieved: int = 0

    # Blocker analysis accuracy: when LLM says X blocks, does fixing X help?
    blockers_identified: int = 0
    blockers_resolved: int = 0

    # State suggestion utility: do suggested variables reduce contradictions?
    variables_suggested: int = 0
    variables_useful: int = 0

    @property
    def recommendation_accuracy(self) -> float:
        if self.recommendations_followed == 0:
            return 0.5  # Prior
        return self.recommendations_helped / self.recommendations_followed

    @property
    def goal_accuracy(self) -> float:
        if self.goals_pursued == 0:
            return 0.5
        return self.goals_achieved / self.goals_pursued

    @property
    def blocker_accuracy(self) -> float:
        if self.blockers_identified == 0:
            return 0.5
        return self.blockers_resolved / self.blockers_identified

    def update_recommendation(self, helped: bool):
        self.recommendations_followed += 1
        if helped:
            self.recommendations_helped += 1

    def get_summary(self) -> Dict:
        return {
            "recommendation_accuracy": self.recommendation_accuracy,
            "goal_accuracy": self.goal_accuracy,
            "blocker_accuracy": self.blocker_accuracy,
            "total_recommendations": self.recommendations_followed,
        }


# =============================================================================
# DYNAMICS MODEL (unchanged from v2)
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

    def observation_count(self, state, action: str) -> int:
        """How many times have we seen this (state, action) pair?"""
        if isinstance(state, GameState):
            key = (state, action)
        else:
            # Accept AgentBeliefs via to_state_key()
            key = (state.to_state_key(), action)
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
# BELIEF UPDATER (v3)
# =============================================================================

class BeliefUpdater:
    """
    Updates agent beliefs based on LLM observations.

    Weights updates by learned reliability.
    """

    def __init__(self, reliability: OracleReliability):
        self.reliability = reliability

    def update_from_understanding(
        self,
        beliefs: AgentBeliefs,
        understanding: SituationUnderstanding,
    ) -> AgentBeliefs:
        """
        Update beliefs based on LLM's situation analysis.

        More reliable sources get more weight.
        """
        # Goal updates - always incorporate (hard to verify, but useful)
        if understanding.overall_goal and beliefs.overall_goal == "unknown":
            beliefs.overall_goal = understanding.overall_goal

        if understanding.immediate_goal:
            beliefs.current_subgoal = understanding.immediate_goal

        # Progress updates - incorporate if confidence is high
        if understanding.confidence > 0.6:
            for item in understanding.accomplished:
                if item not in beliefs.accomplished:
                    beliefs.accomplished.append(item)

        # Blocker updates - very useful for planning
        if understanding.blocking_condition:
            beliefs.blocking_condition = understanding.blocking_condition

        return beliefs

    def update_from_failure_analysis(
        self,
        beliefs: AgentBeliefs,
        action: str,
        analysis: Dict,
    ) -> AgentBeliefs:
        """Update beliefs based on failure analysis."""
        if "failure_reason" in analysis:
            beliefs.failed_actions[action] = analysis["failure_reason"]

        if "prerequisite" in analysis and analysis["prerequisite"]:
            beliefs.blocking_condition = analysis["prerequisite"]

        return beliefs

    def update_from_progress(
        self,
        beliefs: AgentBeliefs,
        progress: Dict,
    ) -> AgentBeliefs:
        """Update beliefs based on progress detection."""
        if progress.get("made_progress") and progress.get("accomplishment"):
            accomplishment = progress["accomplishment"]
            if accomplishment not in beliefs.accomplished:
                beliefs.accomplished.append(accomplishment)

        # Clear blocker if we made progress
        if progress.get("made_progress"):
            beliefs.blocking_condition = None

        return beliefs


# =============================================================================
# ACTION SELECTION (v3)
# =============================================================================

class InformedActionSelector:
    """
    Selects actions using LLM understanding + learned dynamics.

    Oracle is injected — this class remains stdlib-only.
    When oracle is None, falls back to dynamics-only + random exploration.
    """

    def __init__(
        self,
        oracle=None,
        reliability: Optional[OracleReliability] = None,
        dynamics: Optional[DynamicsModel] = None,
        exploration_rate: float = 0.2,
    ):
        self.oracle = oracle
        self.reliability = reliability or OracleReliability()
        self.dynamics = dynamics or DynamicsModel()
        self.exploration_rate = exploration_rate

        # Cache current understanding
        self.current_understanding: Optional[SituationUnderstanding] = None

    def select_action(
        self,
        observation: str,
        beliefs: AgentBeliefs,
        valid_actions: List[str],
    ) -> Tuple[str, str]:
        """
        Select action based on LLM understanding and learned dynamics.

        Returns: (action, reason)
        """
        if not valid_actions:
            return "look", "no valid actions available"

        # Get LLM's situation analysis if oracle available
        if self.oracle is not None:
            understanding = self.oracle.analyse_situation(
                observation, beliefs, valid_actions
            )
            self.current_understanding = understanding
        else:
            understanding = SituationUnderstanding()
            self.current_understanding = understanding

        # Decision hierarchy:

        # 1. If LLM has high-confidence recommendation, probably follow it
        if (understanding.recommended_action and
                understanding.confidence > 0.6 and
                self.reliability.recommendation_accuracy > 0.4):

            rec = understanding.recommended_action.lower()
            for action in valid_actions:
                if action.lower() == rec or rec in action.lower() or action.lower() in rec:
                    # Follow recommendation with probability based on reliability
                    if random.random() < 0.7 + 0.3 * self.reliability.recommendation_accuracy:
                        return action, f"LLM recommended: {understanding.action_reasoning}"

        # 2. If we have a blocker, try to address it
        if understanding.blocking_condition:
            for action in valid_actions:
                if self._action_addresses_blocker(action, understanding.blocking_condition):
                    return action, f"addressing blocker: {understanding.blocking_condition}"

        # 3. Use dynamics model for actions we have experience with
        state_key = beliefs.to_state_key()
        experienced_actions = [
            a for a in valid_actions
            if self.dynamics.observation_count(state_key, a) > 0
        ]

        if experienced_actions and random.random() > self.exploration_rate:
            best_action = max(
                experienced_actions,
                key=lambda a: self.dynamics.expected_reward(state_key, a)
            )
            return best_action, "best learned action"

        # 4. Explore: prefer LLM's alternative suggestions
        if understanding.alternative_actions:
            for alt in understanding.alternative_actions:
                for action in valid_actions:
                    if alt.lower() in action.lower():
                        return action, "exploring LLM alternative"

        # 5. Random exploration
        return random.choice(valid_actions), "random exploration"

    def _action_addresses_blocker(self, action: str, blocker: str) -> bool:
        """Heuristic: does this action seem to address the blocker?"""
        action_lower = action.lower()
        blocker_lower = blocker.lower()

        # Simple keyword matching
        keywords = blocker_lower.split()
        for word in keywords:
            if len(word) > 3 and word in action_lower:
                return True

        # Common patterns
        if "stand" in blocker_lower and "stand" in action_lower:
            return True
        if "dress" in blocker_lower and ("wear" in action_lower or "dress" in action_lower):
            return True
        if "door" in blocker_lower and ("open" in action_lower or "unlock" in action_lower):
            return True

        return False


# =============================================================================
# THE AGENT (v3)
# =============================================================================

class BayesianIFAgent:
    """
    Bayesian agent for interactive fiction with rich LLM oracle integration.

    The LLM helps understand. The agent decides.

    Oracle is injected at construction — core.py never imports oracle.py.
    When oracle is None, falls back to dynamics-only exploration.
    """

    def __init__(
        self,
        oracle=None,
        exploration_rate: float = 0.2,
    ):
        self.beliefs = AgentBeliefs()
        self.reliability = OracleReliability()
        self.dynamics = DynamicsModel()
        self.action_selector = InformedActionSelector(
            oracle=oracle,
            reliability=self.reliability,
            dynamics=self.dynamics,
            exploration_rate=exploration_rate,
        )
        self.belief_updater = BeliefUpdater(self.reliability)

        # Episode tracking
        self.episode_count = 0
        self.previous_observation = ""
        self.previous_action = ""
        self.previous_score = 0

    def reset_episode(self):
        """Reset for new episode, keeping learned knowledge."""
        self.beliefs = AgentBeliefs(
            overall_goal=self.beliefs.overall_goal,  # Keep learned goal
        )
        self.previous_observation = ""
        self.previous_action = ""
        self.previous_score = 0

    def choose_action(self, observation: str, valid_actions: List[str]) -> str:
        """
        Choose action using LLM understanding + Bayesian reasoning.

        Returns the chosen action string.
        """
        action, reason = self.action_selector.select_action(
            observation, self.beliefs, valid_actions
        )

        # Record for learning
        self.previous_observation = observation
        self.previous_action = action

        # Update action history
        self.beliefs.action_history.append(action)
        if len(self.beliefs.action_history) > 50:
            self.beliefs.action_history = self.beliefs.action_history[-50:]

        return action

    def observe_outcome(self, observation: str, reward: float, score: int):
        """
        Learn from outcome.

        Updates beliefs, reliability tracking, and dynamics model.
        """
        score_change = score - self.previous_score

        # Detect progress using oracle
        progress = {"made_progress": False}
        if self.action_selector.oracle is not None and self.previous_action:
            progress = self.action_selector.oracle.detect_progress(
                self.previous_observation,
                self.previous_action,
                observation,
                score_change,
            )

        # Update beliefs from progress
        self.beliefs = self.belief_updater.update_from_progress(
            self.beliefs, progress
        )

        # Update reliability tracking
        understanding = self.action_selector.current_understanding
        if understanding and understanding.recommended_action:
            was_recommended = (
                self.previous_action.lower() in understanding.recommended_action.lower() or
                understanding.recommended_action.lower() in self.previous_action.lower()
            )
            if was_recommended:
                self.reliability.update_recommendation(
                    helped=progress.get("made_progress", False) or score_change > 0
                )

        # Analyse failure if action didn't seem to work
        if (not progress.get("made_progress") and score_change <= 0
                and self.action_selector.oracle is not None and self.previous_action):
            failure_analysis = self.action_selector.oracle.analyse_failure(
                self.previous_observation,
                self.beliefs,
                self.previous_action,
                observation,
            )
            self.beliefs = self.belief_updater.update_from_failure_analysis(
                self.beliefs, self.previous_action, failure_analysis
            )

        # Update dynamics model
        if self.previous_action:
            prev_state_key = self.beliefs.to_state_key()
            # After ground truth update, derive new state key
            new_state_key = self.beliefs.to_state_key()
            self.dynamics.update(
                prev_state_key, self.previous_action, new_state_key,
                reward, observation,
            )

        # Update observation history
        self.beliefs.observation_history.append(observation[:200])
        if len(self.beliefs.observation_history) > 20:
            self.beliefs.observation_history = self.beliefs.observation_history[-20:]

        self.previous_score = score

    def update_beliefs_from_ground_truth(
        self,
        location: str,
        location_id: int,
        inventory: List[str],
        world_hash: str,
    ):
        """Update beliefs from Jericho ground truth."""
        self.beliefs.location = location
        self.beliefs._location_id = location_id
        self.beliefs.inventory = list(inventory)
        self.beliefs._world_hash = world_hash

    def get_statistics(self) -> Dict:
        """Return statistics about the agent's learning."""
        return {
            "episode_count": self.episode_count,
            "transitions_learned": len(self.dynamics.observed_transitions),
            "dynamics_history_size": len(self.dynamics.history),
            "overall_goal": self.beliefs.overall_goal,
            "accomplished": self.beliefs.accomplished,
            "reliability": self.reliability.get_summary(),
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Bayesian IF Agent v3 - Core Unit Tests")
    print("=" * 60)

    # Test 1: GameState
    state1 = GameState(location=5, inventory=frozenset(["keys"]), world_hash="abc123")
    state2 = GameState(location=5, inventory=frozenset(["keys", "wallet"]), world_hash="def456")
    print(f"State 1: {state1}")
    print(f"State 2: {state2}")
    assert state1 != state2
    assert hash(state1) != hash(state2)

    initial = GameState.initial()
    assert initial.location == 0
    assert initial.inventory == frozenset()
    print(f"Initial: {initial}")

    # Test 2: AgentBeliefs
    beliefs = AgentBeliefs(location="bedroom", inventory=["keys", "phone"])
    beliefs._location_id = 5
    beliefs._world_hash = "abc"
    ctx = beliefs.to_prompt_context()
    assert "bedroom" in ctx
    assert "keys" in ctx
    key = beliefs.to_state_key()
    assert key.location == 5
    assert "keys" in key.inventory
    print(f"\nBeliefs context:\n{ctx}")
    print(f"State key: {key}")

    # Test 3: SituationUnderstanding from JSON
    su = SituationUnderstanding.from_json({
        "overall_goal": "escape the house",
        "immediate_goal": "get dressed",
        "recommended_action": "stand up",
        "confidence": 0.8,
    })
    assert su.overall_goal == "escape the house"
    assert su.confidence == 0.8
    print(f"\nUnderstanding: goal={su.overall_goal}, action={su.recommended_action}")

    # Test 4: OracleReliability
    rel = OracleReliability()
    assert rel.recommendation_accuracy == 0.5  # Prior
    for _ in range(7):
        rel.update_recommendation(helped=True)
    for _ in range(3):
        rel.update_recommendation(helped=False)
    assert 0.6 < rel.recommendation_accuracy < 0.8
    print(f"\nReliability: {rel.get_summary()}")

    # Test 5: DynamicsModel (unchanged)
    dynamics = DynamicsModel()
    dynamics.update(state1, "take wallet", state2, 0.0)
    dynamics.update(state1, "take wallet", state2, 0.0)
    pred = dynamics.predict(state1, "take wallet")
    assert len(pred) > 0
    print(f"\nDynamics prediction: {len(pred)} outcomes")

    # Test 6: BeliefUpdater
    updater = BeliefUpdater(rel)
    b = AgentBeliefs()
    su2 = SituationUnderstanding(
        overall_goal="win the game",
        immediate_goal="find the key",
        blocking_condition="door is locked",
        confidence=0.9,
        accomplished=["found map"],
    )
    b = updater.update_from_understanding(b, su2)
    assert b.overall_goal == "win the game"
    assert b.blocking_condition == "door is locked"
    assert "found map" in b.accomplished
    print(f"\nUpdated beliefs: goal={b.overall_goal}, blocker={b.blocking_condition}")

    b = updater.update_from_progress(b, {"made_progress": True, "accomplishment": "opened door"})
    assert b.blocking_condition is None
    assert "opened door" in b.accomplished
    print(f"After progress: blocker={b.blocking_condition}, accomplished={b.accomplished}")

    # Test 7: InformedActionSelector (no oracle)
    selector = InformedActionSelector(dynamics=dynamics, exploration_rate=0.3)
    b2 = AgentBeliefs()
    b2._location_id = 5
    b2._world_hash = "abc123"
    action, reason = selector.select_action("You are in a room.", b2, ["look", "go north", "take wallet"])
    assert action in ["look", "go north", "take wallet"]
    print(f"\nSelected action: {action} ({reason})")

    # Test 8: BayesianIFAgent (no oracle)
    agent = BayesianIFAgent(exploration_rate=0.2)
    agent.update_beliefs_from_ground_truth("bedroom", 1, [], "h0")
    action = agent.choose_action("You wake up.", ["stand up", "sleep", "look"])
    print(f"\nAgent chose: {action}")

    agent.update_beliefs_from_ground_truth("bedroom", 1, [], "h1")
    agent.observe_outcome("You stand up.", 0.0, 1)
    stats = agent.get_statistics()
    print(f"Agent stats: {stats}")

    print("\nAll core v3 tests passed!")
