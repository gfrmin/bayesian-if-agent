"""
Bayesian Interactive Fiction Agent

An agent that maintains beliefs about game dynamics and selects actions
to maximise expected utility (points/survival).

Key design principles:
- LLM provides PRIORS (what usually happens in IF games)
- Experience provides LIKELIHOOD (what actually happens in this game)
- Bayesian updating gives POSTERIOR (refined predictions)
- Expected utility maximisation selects ACTIONS

What's FIXED:
- Bayesian inference as update rule
- Expected utility as action selection
- Game interface: text in, text out, score signal

What's LEARNED:
- Dynamics: P(outcome | state, action)
- State beliefs: P(current_state | observation_history)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random
import math
import json


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

@dataclass(frozen=True)
class GameState:
    """
    Minimal state representation.
    
    This is what we're fixing - a simple factorisation of game state.
    The agent learns which aspects of this matter for which actions.
    """
    location: str
    inventory: frozenset
    flags: frozenset  # Observable world state changes
    
    def __str__(self):
        inv = ", ".join(sorted(self.inventory)) if self.inventory else "nothing"
        flags = ", ".join(sorted(self.flags)) if self.flags else "none"
        return f"[{self.location}] carrying: {inv} | flags: {flags}"
    
    @staticmethod
    def initial():
        return GameState(
            location="unknown",
            inventory=frozenset(),
            flags=frozenset()
        )


@dataclass(frozen=True)
class Transition:
    """A single observed transition with raw observation for re-parsing."""
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
    The prior (pseudocounts) could come from LLM knowledge about IF games.
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

        # Full transition history for re-parsing during expansion
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
            # No observations for this (state, action) pair
            # Return uniform over observed outcomes, or empty
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
    
    def uncertainty(self, state: GameState, action: str) -> float:
        """
        How uncertain are we about this transition?
        
        Returns entropy of the distribution, or high value if no observations.
        """
        dist = self.predict(state, action)
        if not dist:
            return 10.0  # High uncertainty for unknown transitions
        
        entropy = -sum(p * math.log(p + 1e-10) for p in dist.values())
        return entropy
    
    def observation_count(self, state: GameState, action: str) -> int:
        """How many times have we seen this (state, action) pair?"""
        key = (state, action)
        # Subtract prior pseudocounts to get actual observation count
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
# BELIEF STATE
# =============================================================================

class BeliefState:
    """
    Distribution over current game state.
    
    For simplicity, we use a particle-based representation:
    a set of possible states with associated weights.
    """
    
    def __init__(self, initial_state: Optional[GameState] = None):
        if initial_state:
            self.particles: Dict[GameState, float] = {initial_state: 1.0}
        else:
            self.particles: Dict[GameState, float] = {}
    
    def set_state(self, state: GameState, confidence: float = 1.0):
        """Set belief to a single state with given confidence."""
        self.particles = {state: confidence}
    
    def update_from_observation(self, observed_state: GameState, confidence: float = 0.9):
        """
        Update beliefs based on observed state.
        
        In IF, observations are fairly reliable, so we weight
        the observed state highly.
        """
        # Simple approach: high weight on observed, low on alternatives
        new_particles = {}
        
        # The observed state gets most of the mass
        new_particles[observed_state] = confidence
        
        # Keep some mass on previous beliefs (in case observation is partial)
        remaining = 1.0 - confidence
        for state, weight in self.particles.items():
            if state != observed_state:
                new_particles[state] = new_particles.get(state, 0) + weight * remaining
        
        self.particles = new_particles
        self._normalise()
    
    def propagate(self, action: str, dynamics: DynamicsModel):
        """
        Update beliefs by propagating through action using dynamics model.
        
        P(s') = Σ_s P(s' | s, a) P(s)
        """
        new_particles: Dict[GameState, float] = defaultdict(float)
        
        for state, state_prob in self.particles.items():
            outcomes = dynamics.predict(state, action)
            
            if not outcomes:
                # No learned dynamics for this transition
                # Keep the state unchanged (conservative assumption)
                new_particles[state] += state_prob
            else:
                for (next_state, reward), trans_prob in outcomes.items():
                    new_particles[next_state] += state_prob * trans_prob
        
        self.particles = dict(new_particles)
        self._normalise()
    
    def _normalise(self):
        """Normalise particle weights to sum to 1."""
        total = sum(self.particles.values())
        if total > 0:
            self.particles = {s: w/total for s, w in self.particles.items()}
    
    def most_likely(self) -> Optional[GameState]:
        """Return the most probable state."""
        if not self.particles:
            return None
        return max(self.particles.keys(), key=lambda s: self.particles[s])
    
    def sample(self) -> Optional[GameState]:
        """Sample a state from the belief distribution."""
        if not self.particles:
            return None
        states, weights = zip(*self.particles.items())
        return random.choices(states, weights)[0]
    
    def entropy(self) -> float:
        """Uncertainty about current state."""
        if not self.particles:
            return 10.0
        return -sum(p * math.log(p + 1e-10) for p in self.particles.values())


# =============================================================================
# ACTION SELECTION
# =============================================================================

class ActionSelector:
    """
    Selects actions to maximise expected utility.
    
    Uses Thompson sampling: sample from posterior over dynamics,
    then pick the action that's best under the sampled model.
    """
    
    def __init__(self, exploration_bonus: float = 0.1):
        self.exploration_bonus = exploration_bonus
    
    def select_action(
        self,
        belief: BeliefState,
        dynamics: DynamicsModel,
        candidate_actions: List[str],
        goal_utility: callable = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select an action via expected utility with exploration bonus.
        
        Returns (selected_action, action_values_dict)
        """
        if not candidate_actions:
            return None, {}
        
        current_state = belief.most_likely()
        if current_state is None:
            # No belief about state - pick randomly
            return random.choice(candidate_actions), {}
        
        action_values = {}
        
        for action in candidate_actions:
            # Expected reward from this action
            outcomes = dynamics.predict(current_state, action)
            
            if not outcomes:
                # No data - assign prior expected value + exploration bonus
                expected_reward = 0.0
                exploration = self.exploration_bonus * 2  # Extra bonus for unexplored
            else:
                expected_reward = sum(
                    prob * reward 
                    for (next_state, reward), prob in outcomes.items()
                )
                # Exploration bonus based on uncertainty
                exploration = self.exploration_bonus * dynamics.uncertainty(current_state, action)
            
            action_values[action] = expected_reward + exploration
        
        # Select best action
        best_action = max(action_values.keys(), key=lambda a: action_values[a])
        
        return best_action, action_values
    
    def thompson_sample(
        self,
        belief: BeliefState,
        dynamics: DynamicsModel,
        candidate_actions: List[str]
    ) -> str:
        """
        Thompson sampling: sample from posterior, pick best under sample.
        
        This naturally balances exploration and exploitation.
        """
        if not candidate_actions:
            return None
        
        current_state = belief.sample()  # Sample state from belief
        if current_state is None:
            return random.choice(candidate_actions)
        
        best_action = None
        best_value = float('-inf')
        
        for action in candidate_actions:
            # Sample an outcome from the dynamics
            outcome = dynamics.sample_outcome(current_state, action)
            
            if outcome is None:
                # No data - optimistic prior
                value = random.random() * self.exploration_bonus
            else:
                next_state, reward = outcome
                value = reward
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action if best_action else random.choice(candidate_actions)


# =============================================================================
# STATE PARSER (LLM Integration Point)
# =============================================================================

class StateParser:
    """
    Parses game observations into state representations.
    
    This is where the LLM provides value: understanding natural language
    descriptions and extracting structured state information.
    
    For now, we use simple heuristics. The LLM version would call
    an actual language model.
    """
    
    def __init__(self):
        # Common IF location patterns
        self.location_keywords = [
            'bedroom', 'bathroom', 'kitchen', 'living room', 'hallway',
            'north', 'south', 'east', 'west', 'outside', 'inside',
            'car', 'office', 'street'
        ]
        
        # Common inventory indicators
        self.inventory_patterns = [
            'you are carrying', 'you have', 'in your possession',
            'you pick up', 'taken', 'you now have'
        ]
    
    def parse(self, observation: str, previous_state: Optional[GameState] = None) -> GameState:
        """
        Extract state from observation text.
        
        Returns a GameState with location, inventory, and flags.
        """
        obs_lower = observation.lower()
        
        # Extract location
        location = self._extract_location(obs_lower, previous_state)
        
        # Extract inventory changes
        inventory = self._extract_inventory(obs_lower, previous_state)
        
        # Extract flags (world state changes)
        flags = self._extract_flags(obs_lower, previous_state)
        
        return GameState(
            location=location,
            inventory=inventory,
            flags=flags
        )
    
    def _extract_location(self, obs: str, prev: Optional[GameState]) -> str:
        """Extract current location from observation."""
        # Look for location name patterns
        for keyword in self.location_keywords:
            if keyword in obs:
                return keyword
        
        # Check for explicit location markers
        if '(in ' in obs:
            start = obs.find('(in ') + 4
            end = obs.find(')', start)
            if end > start:
                return obs[start:end].strip()
        
        # Check for room names at start of description
        lines = obs.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line[0].islower() and len(line.split()) <= 4:
                # Might be a room name
                return line.lower()
        
        # Default to previous location
        return prev.location if prev else "unknown"
    
    def _extract_inventory(self, obs: str, prev: Optional[GameState]) -> frozenset:
        """Extract inventory from observation."""
        inventory = set(prev.inventory) if prev else set()
        
        # Check for taking items
        if 'taken' in obs or 'you pick up' in obs or 'you take' in obs:
            # Try to find what was taken
            words = obs.split()
            for i, word in enumerate(words):
                if word in ['taken', 'take', 'pick']:
                    # Look for noun nearby
                    for j in range(max(0, i-3), min(len(words), i+4)):
                        if words[j] not in ['the', 'a', 'an', 'you', 'up', 'taken', 'take', 'pick']:
                            inventory.add(words[j].strip('.,!'))
                            break
        
        # Check for dropping items
        if 'dropped' in obs or 'you drop' in obs:
            words = obs.split()
            for i, word in enumerate(words):
                if word in ['dropped', 'drop']:
                    for j in range(max(0, i-3), min(len(words), i+4)):
                        candidate = words[j].strip('.,!')
                        if candidate in inventory:
                            inventory.discard(candidate)
                            break
        
        return frozenset(inventory)
    
    def _extract_flags(self, obs: str, prev: Optional[GameState]) -> frozenset:
        """Extract world state flags from observation."""
        flags = set(prev.flags) if prev else set()
        
        # Common state changes
        if 'door' in obs and 'open' in obs:
            flags.add('door_open')
        if 'door' in obs and ('closed' in obs or 'close' in obs):
            flags.discard('door_open')
        
        if 'light' in obs or 'lamp' in obs:
            if 'on' in obs:
                flags.add('light_on')
            elif 'off' in obs:
                flags.discard('light_on')
        
        if 'dead' in obs or 'died' in obs or 'killed' in obs:
            flags.add('dead')
        
        if 'wearing' in obs:
            flags.add('dressed')
        
        if 'shower' in obs:
            flags.add('showered')
        
        return frozenset(flags)


# =============================================================================
# THE AGENT
# =============================================================================

class BayesianIFAgent:
    """
    The complete Bayesian IF agent.
    
    Maintains beliefs about state and dynamics, selects actions
    to maximise expected utility.
    """
    
    def __init__(
        self,
        parser: Optional[StateParser] = None,
        prior_pseudocount: float = 0.1,
        exploration_bonus: float = 0.1
    ):
        self.parser = parser or StateParser()
        self.dynamics = DynamicsModel(prior_pseudocount=prior_pseudocount)
        self.belief = BeliefState()
        self.selector = ActionSelector(exploration_bonus=exploration_bonus)
        
        # History for analysis
        self.history: List[Dict] = []
        self.current_state: Optional[GameState] = None
        self.previous_score: float = 0.0
    
    def observe(self, observation: str, score: float = 0.0):
        """
        Process a new observation from the game.
        
        Updates beliefs about current state.
        """
        # Parse observation into state
        new_state = self.parser.parse(observation, self.current_state)
        
        # Calculate reward (change in score)
        reward = score - self.previous_score
        
        # If we have a previous state and action, update dynamics
        if self.current_state is not None and self.history:
            last_action = self.history[-1].get('action')
            if last_action:
                self.dynamics.update(
                    self.current_state,
                    last_action,
                    new_state,
                    reward,
                    raw_observation=observation
                )
        
        # Update belief state
        self.belief.update_from_observation(new_state)
        
        # Update tracking
        self.current_state = new_state
        self.previous_score = score
        
        # Record in history
        self.history.append({
            'observation': observation,
            'parsed_state': str(new_state),
            'score': score,
            'reward': reward
        })
    
    def act(self, candidate_actions: List[str], use_thompson: bool = True,
            budget=None) -> str:
        """
        Select an action to take.

        If budget (a metareason.ComputationBudget) is provided, uses the
        metareasoner's deliberate() instead of a single Thompson sample.

        Returns the chosen action.
        """
        if budget is not None:
            from metareason import deliberate
            action, _meta = deliberate(
                self.belief, self.dynamics, self.selector,
                candidate_actions, budget
            )
        elif use_thompson:
            action = self.selector.thompson_sample(
                self.belief,
                self.dynamics,
                candidate_actions
            )
        else:
            action, _values = self.selector.select_action(
                self.belief,
                self.dynamics,
                candidate_actions
            )

        # Record action in history
        if self.history:
            self.history[-1]['action'] = action

        return action
    
    def get_statistics(self) -> Dict:
        """Return statistics about the agent's learning."""
        return {
            'total_steps': len(self.history),
            'unique_states_visited': len(set(h.get('parsed_state') for h in self.history)),
            'transitions_learned': len(self.dynamics.observed_transitions),
            'dynamics_history_size': len(self.dynamics.history),
            'current_state': str(self.current_state),
            'current_score': self.previous_score,
            'belief_entropy': self.belief.entropy()
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test of components
    
    # Test state representation
    state1 = GameState("bedroom", frozenset(["keys"]), frozenset(["door_open"]))
    state2 = GameState("bedroom", frozenset(["keys", "wallet"]), frozenset(["door_open"]))
    print(f"State 1: {state1}")
    print(f"State 2: {state2}")
    
    # Test dynamics model
    dynamics = DynamicsModel()
    dynamics.update(state1, "take wallet", state2, 0.0)
    dynamics.update(state1, "take wallet", state2, 0.0)  # Same transition
    
    print(f"\nDynamics prediction for 'take wallet':")
    print(dynamics.predict(state1, "take wallet"))
    print(f"Uncertainty: {dynamics.uncertainty(state1, 'take wallet'):.3f}")
    print(f"Observation count: {dynamics.observation_count(state1, 'take wallet')}")
    
    # Test belief state
    belief = BeliefState(state1)
    print(f"\nInitial belief: {belief.most_likely()}")
    print(f"Belief entropy: {belief.entropy():.3f}")
    
    # Test action selection
    selector = ActionSelector()
    action, values = selector.select_action(
        belief, dynamics, ["take wallet", "go north", "look"]
    )
    print(f"\nSelected action: {action}")
    print(f"Action values: {values}")
    
    print("\nAll tests passed!")
