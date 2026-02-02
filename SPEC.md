# Adaptive Bayesian IF Agent: Implementation Specification v2

## Overview

An agent that plays interactive fiction games by:
1. Maintaining Bayesian beliefs about game dynamics
2. Using an LLM as a **sensor** (not decision-maker) for action relevance
3. Learning the LLM's reliability from experience
4. Expanding its state representation when needed
5. Deciding how much to deliberate before acting

**Key insight**: Everything is either an observation (to condition on) or a decision (to optimise). The LLM provides observations. The Bayesian machinery makes decisions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          JERICHO                                │
│  Provides: observations, rewards, valid_actions, ground_truth   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM SENSOR (Ollama/Llama3.1)               │
│                                                                 │
│  Input: observation text, valid actions                         │
│  Output: relevance scores (noisy signal of action value)        │
│                                                                 │
│  The agent models this as: P(LLM_output | true_action_value)    │
│  Reliability is LEARNED from experience                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BAYESIAN CORE                              │
│                                                                 │
│  • LLM Sensor Model: P(LLM_score | action_valuable)             │
│  • Dynamics Model: P(next_state, reward | state, action)        │
│  • Action Selection: argmax E[utility] via Thompson sampling    │
│                                                                 │
│  All beliefs are updated via Bayes' rule from observations      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      METAREASONER                               │
│                                                                 │
│  Decides: think more, expand model, or act now?                 │
│  Criterion: V(think) vs V(act)                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. The LLM is a Sensor

The LLM provides **observations**, not decisions.

When the LLM says "answer phone has relevance 0.8", this is data — like a thermometer reading. The agent has beliefs about:
- What the true value of "answer phone" is
- How reliable the LLM's readings are

The agent does NOT simply trust the LLM. It conditions on LLM outputs using a learned sensor model.

### 2. LLM Reliability is Learned

The agent tracks:
- When LLM said an action was relevant, was it actually valuable?
- When LLM said an action was irrelevant, was it actually useless?

This gives estimates of:
- **True positive rate**: P(LLM says relevant | action is valuable)
- **False positive rate**: P(LLM says relevant | action is not valuable)

These rates update with experience. If the LLM is unreliable, the agent learns to ignore it.

### 3. Experience Dominates Over Time

For actions the agent has tried:
- Prior (from LLM) fades
- Posterior (from experience) dominates

For actions the agent hasn't tried:
- Prior (from LLM) guides exploration
- Weighted by learned reliability

### 4. No Reward Shaping

The only reward is game score. No novelty bonuses, no intrinsic rewards.

The LLM helps focus exploration, but doesn't change what the agent is optimising for.

### 5. Simplicity is Cost, Not Prior

We don't say "simple models are probably true."
We say "simple models cost less to compute with."

Model selection is a decision problem, not inference.

---

## LLM Setup: Ollama with Llama 3.1

### Installation

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.1 (8B is good balance of speed/quality)
ollama pull llama3.1:8b

# Or for better quality (slower):
ollama pull llama3.1:70b
```

### Python Interface

```python
import requests
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib

@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1  # Low temperature for consistency
    timeout: int = 30

class OllamaClient:
    """
    Client for local Ollama instance running Llama 3.1.
    """
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.cache: Dict[str, any] = {}
    
    def complete(self, prompt: str, use_cache: bool = True) -> str:
        """
        Get completion from Ollama.
        """
        # Check cache
        cache_key = self._cache_key(prompt)
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Make request
        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "temperature": self.config.temperature,
                "stream": False
            },
            timeout=self.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()["response"]
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = result
        
        return result
    
    def complete_json(self, prompt: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Get JSON completion from Ollama.
        """
        full_prompt = prompt + "\n\nRespond with valid JSON only, no other text."
        
        response = self.complete(full_prompt, use_cache)
        
        # Extract JSON from response
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def clear_cache(self):
        self.cache = {}
```

### Action Relevance Query

```python
class LLMActionSensor:
    """
    Uses LLM to get relevance scores for actions.
    
    This is a SENSOR — its outputs are observations, not decisions.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    def get_relevance_scores(
        self, 
        observation: str, 
        valid_actions: List[str]
    ) -> Dict[str, float]:
        """
        Query LLM for action relevance scores.
        
        Returns: {action: relevance} where relevance ∈ [0, 1]
        """
        # Limit actions to avoid overwhelming the LLM
        actions_to_query = valid_actions[:20]
        
        prompt = f"""You are helping analyse a text adventure game.

Current game state:
{observation}

Available actions:
{json.dumps(actions_to_query, indent=2)}

Rate each action from 0.0 to 1.0 based on how likely it is to advance the story or be useful.

Scoring guide:
- 1.0: Critical action (answering ringing phone, opening exit door, completing objective)
- 0.7-0.9: Likely useful (moving to new area, picking up key item, talking to NPC)
- 0.4-0.6: Possibly useful (examining things, trying doors)
- 0.1-0.3: Unlikely useful (rearranging inventory, looking at mundane objects)
- 0.0: Definitely useless (repeating failed action, nonsensical command)

Return a JSON object mapping each action to its score."""

        result = self.client.complete_json(prompt)
        
        if result is None:
            # Fallback: uniform scores
            return {a: 0.5 for a in actions_to_query}
        
        # Normalise and fill in missing actions
        scores = {}
        for action in valid_actions:
            if action in result:
                # Clamp to [0, 1]
                scores[action] = max(0.0, min(1.0, float(result[action])))
            else:
                # Default for actions not scored
                scores[action] = 0.3
        
        return scores
```

---

## LLM Sensor Model

The agent models the relationship between LLM outputs and true action values.

```python
from dataclasses import dataclass, field
from typing import List, Tuple
import math

@dataclass
class LLMSensorModel:
    """
    Models the reliability of the LLM sensor.
    
    The agent learns:
    - P(LLM says relevant | action is actually valuable)  [true positive rate]
    - P(LLM says relevant | action is not valuable)       [false positive rate]
    
    These are updated from experience.
    """
    
    # Prior beliefs (will be updated)
    # Start optimistic: assume LLM is moderately reliable
    true_positive_alpha: float = 7.0   # Beta prior: ~70% true positive rate
    true_positive_beta: float = 3.0
    
    false_positive_alpha: float = 2.0  # Beta prior: ~20% false positive rate  
    false_positive_beta: float = 8.0
    
    # Threshold for "LLM says relevant"
    relevance_threshold: float = 0.5
    
    # Threshold for "action was valuable"
    value_threshold: float = 0.0  # Any positive reward counts as valuable
    
    @property
    def true_positive_rate(self) -> float:
        """E[P(LLM relevant | valuable)]"""
        return self.true_positive_alpha / (self.true_positive_alpha + self.true_positive_beta)
    
    @property
    def false_positive_rate(self) -> float:
        """E[P(LLM relevant | not valuable)]"""
        return self.false_positive_alpha / (self.false_positive_alpha + self.false_positive_beta)
    
    def update(self, llm_score: float, actual_reward: float):
        """
        Update beliefs about LLM reliability based on observed outcome.
        
        Args:
            llm_score: What the LLM said (0-1)
            actual_reward: What reward the action actually got
        """
        llm_said_relevant = llm_score >= self.relevance_threshold
        action_was_valuable = actual_reward > self.value_threshold
        
        if action_was_valuable:
            # Update true positive rate
            if llm_said_relevant:
                self.true_positive_alpha += 1  # True positive
            else:
                self.true_positive_beta += 1   # False negative
        else:
            # Update false positive rate
            if llm_said_relevant:
                self.false_positive_alpha += 1  # False positive
            else:
                self.false_positive_beta += 1   # True negative
    
    def likelihood(self, llm_score: float, action_valuable: bool) -> float:
        """
        P(LLM gives this score | action has this value)
        
        Models LLM score as drawn from Beta distribution.
        """
        if action_valuable:
            # LLM should give high scores to valuable actions
            # Use true positive rate as mode of Beta distribution
            a = self.true_positive_alpha
            b = self.true_positive_beta
        else:
            # LLM might give high scores to non-valuable actions
            a = self.false_positive_alpha
            b = self.false_positive_beta
        
        # Beta PDF (unnormalised is fine for likelihood ratios)
        if llm_score <= 0 or llm_score >= 1:
            llm_score = max(0.01, min(0.99, llm_score))
        
        return (llm_score ** (a - 1)) * ((1 - llm_score) ** (b - 1))
    
    def posterior_valuable(self, llm_score: float, prior_valuable: float = 0.1) -> float:
        """
        P(action valuable | LLM score) via Bayes' rule.
        
        Args:
            llm_score: The LLM's relevance score
            prior_valuable: Prior probability any action is valuable
        
        Returns:
            Posterior probability the action is valuable
        """
        # P(valuable | score) ∝ P(score | valuable) × P(valuable)
        p_score_if_valuable = self.likelihood(llm_score, True)
        p_score_if_not = self.likelihood(llm_score, False)
        
        numerator = p_score_if_valuable * prior_valuable
        denominator = (
            p_score_if_valuable * prior_valuable +
            p_score_if_not * (1 - prior_valuable)
        )
        
        if denominator == 0:
            return prior_valuable
        
        return numerator / denominator
    
    def get_statistics(self) -> Dict:
        """Return summary statistics for logging."""
        return {
            "true_positive_rate": self.true_positive_rate,
            "false_positive_rate": self.false_positive_rate,
            "tp_observations": self.true_positive_alpha + self.true_positive_beta - 10,
            "fp_observations": self.false_positive_alpha + self.false_positive_beta - 10,
        }
```

---

## Dynamics Model

Learns P(next_state, reward | state, action) from experience.

```python
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Tuple, Set, List, Optional, Any
import random
import math

@dataclass(frozen=True)
class GameState:
    """
    Agent's representation of game state.
    
    Using Jericho ground truth for Phase 1.
    """
    location: int  # Location ID from Jericho
    inventory: frozenset  # Item names
    world_hash: str  # Full state hash for disambiguation
    
    def __str__(self):
        return f"State(loc={self.location}, inv={len(self.inventory)} items)"

@dataclass(frozen=True)
class Outcome:
    """Result of taking an action."""
    next_state: GameState
    reward: float

class DynamicsModel:
    """
    Learns game dynamics from experience.
    
    Uses count-based model with Dirichlet prior.
    Detects contradictions (indicates state representation is insufficient).
    """
    
    def __init__(self, pseudocount: float = 0.1):
        self.pseudocount = pseudocount
        
        # (state, action) -> {outcome: count}
        self.counts: Dict[Tuple[GameState, str], Dict[Outcome, float]] = defaultdict(
            lambda: defaultdict(lambda: self.pseudocount)
        )
        
        # Track observation counts separately
        self.observation_counts: Dict[Tuple[GameState, str], int] = defaultdict(int)
        
        # Track contradictions
        self.contradictions: List[Tuple[GameState, str, Set[Outcome]]] = []
        
        # Full history for re-parsing
        self.history: List[Tuple[GameState, str, Outcome, str]] = []  # (state, action, outcome, raw_obs)
    
    def update(self, state: GameState, action: str, outcome: Outcome, raw_observation: str = ""):
        """
        Record an observed transition.
        """
        key = (state, action)
        
        # Check for contradiction (same state+action, different outcome)
        existing_outcomes = {o for o, c in self.counts[key].items() if c > self.pseudocount}
        if existing_outcomes and outcome not in existing_outcomes:
            self.contradictions.append((state, action, existing_outcomes | {outcome}))
        
        # Update counts
        self.counts[key][outcome] += 1
        self.observation_counts[key] += 1
        
        # Store history
        self.history.append((state, action, outcome, raw_observation))
    
    def predict(self, state: GameState, action: str) -> Dict[Outcome, float]:
        """
        P(outcome | state, action) as normalised counts.
        """
        key = (state, action)
        counts = self.counts[key]
        total = sum(counts.values())
        
        if total == 0:
            return {}
        
        return {outcome: count / total for outcome, count in counts.items()}
    
    def expected_reward(self, state: GameState, action: str) -> float:
        """
        E[reward | state, action]
        """
        dist = self.predict(state, action)
        if not dist:
            return 0.0
        return sum(prob * outcome.reward for outcome, prob in dist.items())
    
    def sample_outcome(self, state: GameState, action: str) -> Optional[Outcome]:
        """
        Sample from P(outcome | state, action) for Thompson sampling.
        """
        dist = self.predict(state, action)
        if not dist:
            return None
        
        outcomes, probs = zip(*dist.items())
        return random.choices(outcomes, probs)[0]
    
    def uncertainty(self, state: GameState, action: str) -> float:
        """
        Entropy of outcome distribution. High = uncertain.
        """
        dist = self.predict(state, action)
        if not dist:
            return float('inf')  # Maximum uncertainty for unknown
        
        entropy = 0.0
        for prob in dist.values():
            if prob > 0:
                entropy -= prob * math.log(prob)
        
        return entropy
    
    def observation_count(self, state: GameState, action: str) -> int:
        """How many times have we seen this (state, action)?"""
        return self.observation_counts[(state, action)]
    
    def get_contradictions(self) -> List:
        """Return detected contradictions."""
        return self.contradictions
    
    def clear_contradictions(self):
        """Clear contradiction list (after handling them)."""
        self.contradictions = []
    
    def get_statistics(self) -> Dict:
        """Return summary statistics."""
        total_transitions = sum(self.observation_counts.values())
        unique_state_actions = len(self.observation_counts)
        
        return {
            "total_transitions": total_transitions,
            "unique_state_actions": unique_state_actions,
            "contradictions": len(self.contradictions),
        }
```

---

## Action Selection

Combines LLM prior with learned dynamics.

```python
class BayesianActionSelector:
    """
    Selects actions using:
    - Prior from LLM (treated as noisy observation)
    - Learned dynamics (from experience)
    - Learned LLM reliability
    """
    
    def __init__(
        self,
        llm_sensor: LLMActionSensor,
        sensor_model: LLMSensorModel,
        dynamics: DynamicsModel,
        base_prior: float = 0.1,  # Prior P(action is valuable) before LLM
        exploration_weight: float = 0.1,
    ):
        self.llm_sensor = llm_sensor
        self.sensor_model = sensor_model
        self.dynamics = dynamics
        self.base_prior = base_prior
        self.exploration_weight = exploration_weight
        
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
        self.current_llm_scores = self.llm_sensor.get_relevance_scores(
            observation, valid_actions
        )
        
        # Compute value for each action
        action_values = {}
        
        for action in valid_actions:
            llm_score = self.current_llm_scores.get(action, 0.5)
            n_obs = self.dynamics.observation_count(state, action)
            
            if n_obs > 0:
                # We have experience: blend learned value with fading prior
                learned_value = self.dynamics.expected_reward(state, action)
                prior_value = self.llm_score_to_value(llm_score)
                
                # Prior weight decays with experience
                prior_weight = 1.0 / (1.0 + n_obs)
                experience_weight = 1.0 - prior_weight
                
                value = experience_weight * learned_value + prior_weight * prior_value
                
                # Add exploration bonus for uncertainty
                uncertainty = self.dynamics.uncertainty(state, action)
                value += self.exploration_weight * uncertainty
                
            else:
                # No experience: use LLM-informed prior
                value = self.llm_score_to_value(llm_score)
                
                # High exploration bonus for untried actions
                value += self.exploration_weight * 2.0
            
            action_values[action] = value
        
        # Thompson sampling: add noise, pick best
        return self.thompson_sample(action_values)
    
    def llm_score_to_value(self, llm_score: float) -> float:
        """
        Convert LLM score to expected value using sensor model.
        
        P(valuable | LLM_score) via Bayes, then expected value.
        """
        p_valuable = self.sensor_model.posterior_valuable(
            llm_score, self.base_prior
        )
        
        # Expected value: P(valuable) × value_if_valuable + P(not) × 0
        # Assume valuable actions give ~1 unit of reward on average
        expected_value_if_valuable = 1.0
        
        return p_valuable * expected_value_if_valuable
    
    def thompson_sample(self, action_values: Dict[str, float]) -> str:
        """
        Thompson sampling: add noise proportional to uncertainty, pick best.
        """
        noisy_values = {}
        
        for action, value in action_values.items():
            # Add Gaussian noise
            noise = random.gauss(0, 0.1)
            noisy_values[action] = value + noise
        
        return max(noisy_values, key=noisy_values.get)
    
    def observe_outcome(self, state: GameState, action: str, outcome: Outcome):
        """
        Learn from outcome.
        Update both dynamics AND LLM sensor model.
        """
        # Update dynamics
        self.dynamics.update(state, action, outcome)
        
        # Update LLM sensor model
        llm_score = self.current_llm_scores.get(action, 0.5)
        self.sensor_model.update(llm_score, outcome.reward)
    
    def get_llm_score(self, action: str) -> float:
        """Get cached LLM score for action."""
        return self.current_llm_scores.get(action, 0.5)
```

---

## Metareasoning

Decides how much to deliberate before acting.

```python
from dataclasses import dataclass
import time

@dataclass
class MetaState:
    """State of deliberation."""
    best_action: Optional[str] = None
    best_value: float = float('-inf')
    decision_entropy: float = float('inf')
    computation_steps: int = 0

class MetaReasoner:
    """
    Decides when to stop thinking and act.
    
    V(think more) = P(improves) × improvement - time_cost
    V(act now) = current best value
    
    Stop when V(act) > V(think)
    """
    
    def __init__(self, time_cost: float = 0.1):
        self.time_cost = time_cost
    
    def should_continue(self, meta_state: MetaState, budget_remaining: float) -> bool:
        """Should we keep deliberating?"""
        if budget_remaining <= 0:
            return False
        
        if meta_state.best_action is None:
            return True  # Haven't even started
        
        # Probability more thinking improves decision (diminishing returns)
        p_improves = meta_state.decision_entropy * (0.5 ** meta_state.computation_steps)
        
        # Expected improvement
        expected_improvement = meta_state.decision_entropy * 0.1
        
        # Value of more thinking
        v_think = p_improves * expected_improvement - self.time_cost
        
        return v_think > 0 and meta_state.decision_entropy > 0.05
    
    def compute_entropy(self, values: List[float]) -> float:
        """Entropy of softmax distribution over values."""
        if not values or len(values) < 2:
            return 0.0
        
        # Softmax
        max_v = max(values)
        exp_values = [math.exp(v - max_v) for v in values]
        total = sum(exp_values)
        probs = [e / total for e in exp_values]
        
        # Entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
```

---

## Main Agent

Ties everything together.

```python
from jericho import FrotzEnv
from typing import List, Dict, Optional
import time

class BayesianIFAgent:
    """
    Bayesian agent for interactive fiction.
    
    Uses LLM as sensor for action relevance.
    Learns dynamics and LLM reliability from experience.
    """
    
    def __init__(
        self,
        ollama_config: OllamaConfig = None,
        exploration_weight: float = 0.1,
        deliberation_budget: float = 1.0,
    ):
        # LLM setup
        self.llm_client = OllamaClient(ollama_config or OllamaConfig())
        self.llm_sensor = LLMActionSensor(self.llm_client)
        
        # Bayesian components
        self.sensor_model = LLMSensorModel()
        self.dynamics = DynamicsModel()
        
        # Action selection
        self.action_selector = BayesianActionSelector(
            llm_sensor=self.llm_sensor,
            sensor_model=self.sensor_model,
            dynamics=self.dynamics,
            exploration_weight=exploration_weight,
        )
        
        # Metareasoning
        self.meta_reasoner = MetaReasoner()
        self.deliberation_budget = deliberation_budget
        
        # State tracking
        self.current_state: Optional[GameState] = None
        self.episode_count = 0
        self.total_steps = 0
    
    def extract_state(self, env: FrotzEnv) -> GameState:
        """
        Extract state from Jericho environment.
        
        Uses ground truth from Jericho.
        """
        location = env.get_player_location()
        location_id = location.num if location else 0
        
        inventory = env.get_inventory()
        inventory_set = frozenset(obj.name for obj in inventory) if inventory else frozenset()
        
        world_hash = str(env.get_world_state_hash())
        
        return GameState(
            location=location_id,
            inventory=inventory_set,
            world_hash=world_hash,
        )
    
    def choose_action(self, env: FrotzEnv, observation: str) -> str:
        """
        Choose action with bounded deliberation.
        """
        state = self.extract_state(env)
        self.current_state = state
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return "look"
        
        # Select action (includes LLM query)
        action = self.action_selector.select_action(
            state, valid_actions, observation
        )
        
        return action
    
    def observe_outcome(self, env: FrotzEnv, action: str, observation: str, reward: float):
        """
        Learn from outcome.
        """
        if self.current_state is None:
            return
        
        next_state = self.extract_state(env)
        outcome = Outcome(next_state=next_state, reward=reward)
        
        # Update action selector (dynamics + sensor model)
        self.action_selector.observe_outcome(self.current_state, action, outcome)
        
        self.total_steps += 1
    
    def play_episode(self, env: FrotzEnv, max_steps: int = 100, verbose: bool = False) -> Dict:
        """
        Play one episode.
        """
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {self.episode_count + 1}")
            print(f"{'='*60}")
            print(obs[:500])
        
        while not env.game_over() and steps < max_steps:
            # Choose action
            action = self.choose_action(env, obs)
            
            if verbose:
                llm_score = self.action_selector.get_llm_score(action)
                print(f"\n> {action} (LLM relevance: {llm_score:.2f})")
            
            # Take action
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if verbose and reward > 0:
                print(f"  *** REWARD: {reward} ***")
            if verbose:
                print(obs[:300])
            
            # Learn from outcome
            self.observe_outcome(env, action, obs, reward)
        
        self.episode_count += 1
        
        return {
            "episode": self.episode_count,
            "total_reward": total_reward,
            "steps": steps,
            "dynamics_stats": self.dynamics.get_statistics(),
            "sensor_stats": self.sensor_model.get_statistics(),
        }
    
    def play_multiple_episodes(
        self,
        game_path: str,
        n_episodes: int = 10,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Play multiple episodes, learning across them.
        """
        env = FrotzEnv(game_path)
        results = []
        
        for i in range(n_episodes):
            result = self.play_episode(env, max_steps, verbose=(verbose and i < 3))
            results.append(result)
            
            print(f"Episode {i+1}: reward={result['total_reward']}, "
                  f"steps={result['steps']}, "
                  f"transitions={result['dynamics_stats']['total_transitions']}")
        
        # Summary
        rewards = [r['total_reward'] for r in results]
        print(f"\nSummary: mean_reward={sum(rewards)/len(rewards):.2f}, "
              f"max_reward={max(rewards)}")
        print(f"LLM reliability: TPR={self.sensor_model.true_positive_rate:.2f}, "
              f"FPR={self.sensor_model.false_positive_rate:.2f}")
        
        return results
```

---

## Runner Script

```python
#!/usr/bin/env python3
"""
Run the Bayesian IF agent.

Usage:
    python runner.py [game_path] [--episodes N] [--verbose]
"""

import argparse
from agent import BayesianIFAgent, OllamaConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", nargs="?", default="games/905.z5")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()
    
    # Check Ollama is running
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except:
        print("ERROR: Ollama not running. Start it with: ollama serve")
        return
    
    # Create agent
    config = OllamaConfig(model=args.model)
    agent = BayesianIFAgent(ollama_config=config)
    
    # Play
    results = agent.play_multiple_episodes(
        game_path=args.game,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    
    # Save results
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to results.json")

if __name__ == "__main__":
    main()
```

---

## Testing

### Test 1: LLM sensor returns sensible scores

```python
def test_llm_sensor():
    client = OllamaClient()
    sensor = LLMActionSensor(client)
    
    observation = "The phone is ringing loudly. You are lying in bed."
    actions = ["answer phone", "put sock on bed", "sleep", "stand up", "look"]
    
    scores = sensor.get_relevance_scores(observation, actions)
    
    # "answer phone" should score higher than "put sock on bed"
    assert scores["answer phone"] > scores["put sock on bed"]
    print("LLM sensor test passed")
    print(f"Scores: {scores}")
```

### Test 2: Sensor model learns reliability

```python
def test_sensor_model_learning():
    model = LLMSensorModel()
    
    # Simulate: LLM is 80% accurate on valuable actions
    for _ in range(100):
        if random.random() < 0.8:
            model.update(llm_score=0.8, actual_reward=1.0)  # True positive
        else:
            model.update(llm_score=0.2, actual_reward=1.0)  # False negative
    
    # Should learn high true positive rate
    assert model.true_positive_rate > 0.7
    print(f"Learned TPR: {model.true_positive_rate:.2f}")
```

### Test 3: Experience dominates prior

```python
def test_experience_dominates():
    # ... setup agent ...
    
    # Action that LLM rates highly but is actually useless
    llm_score = 0.9
    actual_rewards = [0, 0, 0, 0, 0]  # Never gives reward
    
    # Initially, prior dominates
    initial_value = selector.llm_score_to_value(llm_score)
    
    # After experience, learned value should dominate
    for reward in actual_rewards:
        selector.observe_outcome(state, action, Outcome(state, reward))
    
    # Value should have decreased
    # (would need to track action-specific values to test properly)
```

---

## File Structure

```
bayesian_if_agent/
├── core/
│   ├── __init__.py
│   ├── state.py          # GameState
│   ├── dynamics.py       # DynamicsModel
│   ├── sensor_model.py   # LLMSensorModel
│   ├── action_selector.py # BayesianActionSelector
│   └── metareason.py     # MetaReasoner
├── llm/
│   ├── __init__.py
│   ├── ollama_client.py  # OllamaClient
│   └── action_sensor.py  # LLMActionSensor
├── agent.py              # BayesianIFAgent
├── runner.py             # CLI runner
├── tests/
│   ├── test_sensor.py
│   ├── test_dynamics.py
│   └── test_integration.py
├── games/
│   └── 905.z5
├── requirements.txt
└── README.md
```

---

## Requirements

```
jericho>=3.0.0
requests>=2.28.0
```

---

## Setup Checklist

1. Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
2. Pull model: `ollama pull llama3.1:8b`
3. Start Ollama: `ollama serve` (runs on localhost:11434)
4. Install Python deps: `pip install -r requirements.txt`
5. Download game file to `games/905.z5`
6. Run: `python runner.py --verbose`

---

## Key Metrics to Track

1. **Score per episode** — is the agent improving?
2. **LLM true positive rate** — is the LLM actually helpful?
3. **LLM false positive rate** — how often does LLM mislead?
4. **Unique states visited** — is exploration improving?
5. **Actions with experience** — is coverage growing?

---

## Success Criteria

1. **LLM sensor works**: High-relevance actions (answer phone) score higher than low-relevance (put sock on bed)
2. **Reliability learning works**: Agent learns to trust/distrust LLM based on outcomes
3. **Exploration improves**: Agent reaches more rooms over episodes than random baseline
4. **Score improves**: Later episodes score higher than earlier ones
5. **Principled**: All updates follow Bayesian logic, LLM is sensor not oracle
