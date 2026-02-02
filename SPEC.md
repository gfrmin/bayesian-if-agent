# Adaptive Bayesian IF Agent: Specification v4

## The Core Insight

The LLM is a **queryable sensor bank**. We can ask it anything. To use its answers as proper Bayesian evidence, we restrict outputs to simple forms (yes/no) and learn the sensor's reliability from experience.

The agent is an **expected utility maximiser**. Every action choice is argmax E[U]. No hacks, no overrides, no "follow the oracle X% of the time."

Direct experience dominates. In a deterministic game, once you've observed an outcome, there's no uncertainty. The LLM's opinion becomes irrelevant for that state-action pair.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                              GAME                                   │
│                                                                     │
│  Observations: text, score, valid_actions                           │
│  Ground truth: outcomes of actions (deterministic)                  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM SENSOR BANK                                │
│                                                                     │
│  Binary questions with learned reliability:                         │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │ "Will X help?"      │  │ "Am I in location?" │                   │
│  │ TPR: 0.65 FPR: 0.30 │  │ TPR: 0.85 FPR: 0.10 │                   │
│  └─────────────────────┘  └─────────────────────┘                   │
│                                                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐                   │
│  │ "Do I have item?"   │  │ "Is goal done?"     │                   │
│  │ TPR: 0.80 FPR: 0.15 │  │ TPR: 0.60 FPR: 0.25 │                   │
│  └─────────────────────┘  └─────────────────────┘                   │
│                                                                     │
│  Agent can ask ANY question. Learns reliability from ground truth.  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BELIEF STATE                                   │
│                                                                     │
│  P(in_bedroom) = 0.9                                                │
│  P(have_keys) = 0.3                                                 │
│  P(phone_answered) = 0.8                                            │
│  P(action_X_helps) = 0.6                                            │
│  ...                                                                │
│                                                                     │
│  Updated via Bayes' rule from:                                      │
│  - LLM sensor readings (weighted by learned reliability)            │
│  - Direct observations from game                                    │
│  - Action outcomes (deterministic → certainty)                      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DYNAMICS MODEL                                 │
│                                                                     │
│  For each (state, action) we've tried:                              │
│  - Observed outcome (next_state, reward)                            │
│  - Observation count                                                │
│                                                                     │
│  Deterministic game → 1 observation = certainty                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EXPECTED UTILITY MAXIMISER                     │
│                                                                     │
│  For each action:                                                   │
│    If observed before: E[U] = observed_reward (certain)             │
│    If not observed: E[U] = belief_weighted_estimate + info_value    │
│                                                                     │
│  Choose: argmax E[U]                                                │
│                                                                     │
│  No hacks. No overrides. Just expected utility.                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. The LLM is a Sensor with Learnable Reliability

We can ask the LLM any yes/no question. Its answers are evidence, not truth.

For each question type, we learn:
- **TPR**: P(LLM says Yes | actually Yes)
- **FPR**: P(LLM says Yes | actually No)

We learn these from ground truth:
- Direct game feedback ("You're in the kitchen" when LLM said bedroom)
- Action outcomes (LLM said "X will help", we tried X, it didn't)

### 2. Direct Experience is Certain

The game is deterministic. If we tried action A in state S and got outcome O, we **know** that's what happens. No uncertainty. No need to ask the LLM.

P(outcome = O | state S, action A, already observed O) = 1.0

### 3. Every Decision is Expected Utility Maximisation

No special cases. No "follow oracle 70% of time". No "loop detection".

```
action* = argmax_a E[U | state, action, all_evidence]
```

Where all_evidence includes:
- Direct observations from game
- Previous action outcomes  
- LLM sensor readings (weighted by reliability)

### 4. Unified Decision Space

The agent chooses between two types of actions:
- `take(A)` — take game action A, receive reward, advance game
- `ask(Q)` — ask question Q to the LLM, update beliefs, no game reward

Both are evaluated in the same framework:

**For game actions:**
$$\mathbb{E}[U | \text{take}(A)] = \mathbb{E}[R | A, \text{beliefs}]$$

**For questions:**
$$\mathbb{E}[U | \text{ask}(Q)] = \text{VOI}(Q) - c$$

Where VOI(Q) is the expected improvement in subsequent action value from learning the answer, and $c$ is the cost of asking.

**The decision:**
$$\text{action}^* = \arg\max \left( \max_A \mathbb{E}[U | \text{take}(A)], \max_Q \mathbb{E}[U | \text{ask}(Q)] \right)$$

If a question's VOI minus cost exceeds all game actions' EU, ask first. Otherwise, act.

### 5. The Cost of Asking

The cost $c$ is real, not a hack. It represents the value of computation time.

Even with a local LLM, asking takes wall-clock time. The agent exists in the real world where time has value. The parameter $c$ answers: "how much reward would I trade to save one LLM query?"

- $c = 0$: Ask until VOI = 0 (unbounded questions if VOI never exactly zero)
- $c > 0$: Ask less, act sooner
- $c$ large: Almost never ask, rely on priors and experience

The right $c$ depends on context. For a game we'll play many times, $c$ can be small (information is valuable). For a one-shot situation, $c$ might be larger.

### 6. No Exploration Bonus Needed

For untried actions, expected utility comes from beliefs:

$$\mathbb{E}[U | \text{untried}] = P(\text{helps}) \times R_{\text{success}} + P(\text{doesn't}) \times R_{\text{fail}}$$

For tried actions in a deterministic game, expected utility is the known outcome:

$$\mathbb{E}[U | \text{tried}] = R_{\text{observed}}$$

If we tried an action and got 0, its EU is 0. If we haven't tried an action and believe P(helps) = 0.5, its EU is 0.5. The untried action wins.

**No exploration bonus needed.** Uncertainty itself gives higher expected value (when there's a chance of positive reward). The agent naturally explores because unknown actions have higher expected utility than known-bad actions.

---

## Data Structures

### Binary Sensor

```python
from dataclasses import dataclass
import math

@dataclass
class BinarySensor:
    """
    A yes/no sensor with learned reliability.
    
    Models P(sensor_output | truth) as Beta distributions.
    """
    
    # True positive: P(says Yes | actually Yes)
    tp_alpha: float = 7.0   # Prior: ~70% TPR
    tp_beta: float = 3.0
    
    # False positive: P(says Yes | actually No)
    fp_alpha: float = 3.0   # Prior: ~30% FPR
    fp_beta: float = 7.0
    
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
        """Overall reliability score."""
        return self.tpr - self.fpr  # Ranges from -1 to 1
    
    def update(self, said_yes: bool, was_true: bool):
        """
        Update reliability from ground truth.
        
        Called when we discover the actual truth.
        """
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
            # P(true | yes) = P(yes | true) P(true) / P(yes)
            p_yes_if_true = self.tpr
            p_yes_if_false = self.fpr
            
            numerator = p_yes_if_true * prior
            denominator = p_yes_if_true * prior + p_yes_if_false * (1 - prior)
        else:
            # P(true | no) = P(no | true) P(true) / P(no)
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
```

### Sensor Bank

```python
from typing import Dict, Tuple, Optional
from enum import Enum

class QuestionType(Enum):
    ACTION_HELPS = "action_helps"       # "Will action X make progress?"
    IN_LOCATION = "in_location"         # "Am I in the bedroom?"
    HAVE_ITEM = "have_item"             # "Do I have the keys?"
    STATE_FLAG = "state_flag"           # "Is the door locked?"
    GOAL_DONE = "goal_done"             # "Have I answered the phone?"
    PREREQ_MET = "prereq_met"           # "Can I go east?"
    ACTION_POSSIBLE = "action_possible" # "Is 'open door' available?"

class LLMSensorBank:
    """
    The LLM as a collection of binary sensors.
    
    Each question type has independently learned reliability.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
        
        # Initialize sensors with different priors based on expected reliability
        self.sensors: Dict[QuestionType, BinarySensor] = {
            QuestionType.ACTION_HELPS: BinarySensor(tp_alpha=6, tp_beta=4, fp_alpha=3, fp_beta=7),
            QuestionType.IN_LOCATION: BinarySensor(tp_alpha=8, tp_beta=2, fp_alpha=1, fp_beta=9),
            QuestionType.HAVE_ITEM: BinarySensor(tp_alpha=8, tp_beta=2, fp_alpha=1, fp_beta=9),
            QuestionType.STATE_FLAG: BinarySensor(tp_alpha=6, tp_beta=4, fp_alpha=2, fp_beta=8),
            QuestionType.GOAL_DONE: BinarySensor(tp_alpha=6, tp_beta=4, fp_alpha=2, fp_beta=8),
            QuestionType.PREREQ_MET: BinarySensor(tp_alpha=5, tp_beta=5, fp_alpha=3, fp_beta=7),
            QuestionType.ACTION_POSSIBLE: BinarySensor(tp_alpha=7, tp_beta=3, fp_alpha=2, fp_beta=8),
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
        
        Returns: (answer, posterior_multiplier)
        
        The posterior_multiplier indicates how much to update beliefs:
        - High reliability sensor saying yes → large positive update
        - Low reliability sensor → small update
        """
        cache_key = f"{question_type.value}:{question}"
        
        if cache_key in self.query_cache:
            return self.query_cache[cache_key], self.sensors[question_type].reliability
        
        # Query the LLM
        answer = self._query_llm(question, context)
        
        # Update query count
        self.sensors[question_type].query_count += 1
        
        # Cache result
        self.query_cache[cache_key] = answer
        
        return answer, self.sensors[question_type].reliability
    
    def update_from_ground_truth(
        self, 
        question_type: QuestionType, 
        said_yes: bool, 
        actual_truth: bool,
    ):
        """
        Update sensor reliability from observed ground truth.
        """
        self.sensors[question_type].update(said_yes, actual_truth)
    
    def get_posterior(
        self,
        question_type: QuestionType,
        prior: float,
        said_yes: bool,
    ) -> float:
        """
        Compute posterior probability given sensor reading.
        """
        return self.sensors[question_type].posterior(prior, said_yes)
    
    def clear_cache(self):
        """Clear query cache (call at start of each turn)."""
        self.query_cache = {}
    
    def _query_llm(self, question: str, context: str) -> bool:
        """
        Query LLM for yes/no answer.
        """
        prompt = f"""Answer this question about a text adventure game with only YES or NO.

Context:
{context}

Question: {question}

Answer (YES or NO):"""
        
        response = self.llm.complete(prompt).strip().upper()
        
        return response.startswith("YES")
    
    def get_all_stats(self) -> Dict[str, dict]:
        return {qt.value: sensor.get_stats() for qt, sensor in self.sensors.items()}
```

### Belief State

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional

@dataclass
class BeliefState:
    """
    Agent's beliefs as probability distributions.
    
    Each belief is a probability that something is true.
    Updated via Bayes' rule from sensor readings and direct observations.
    """
    
    # Location belief: P(in_location_X) for each location
    location_beliefs: Dict[str, float] = field(default_factory=dict)
    current_location: Optional[str] = None  # Most likely location
    
    # Inventory beliefs: P(have_item_X)
    inventory_beliefs: Dict[str, float] = field(default_factory=dict)
    
    # State flag beliefs: P(flag_X_is_true)
    flag_beliefs: Dict[str, float] = field(default_factory=dict)
    
    # Goal beliefs: P(goal_X_accomplished)
    goal_beliefs: Dict[str, float] = field(default_factory=dict)
    
    # Action value beliefs: P(action_X_will_help)
    action_beliefs: Dict[str, float] = field(default_factory=dict)
    
    def update_belief(self, belief_name: str, category: str, new_probability: float):
        """Update a belief to a new probability."""
        if category == "location":
            self.location_beliefs[belief_name] = new_probability
            # Update current_location to most likely
            if self.location_beliefs:
                self.current_location = max(self.location_beliefs, key=self.location_beliefs.get)
        elif category == "inventory":
            self.inventory_beliefs[belief_name] = new_probability
        elif category == "flag":
            self.flag_beliefs[belief_name] = new_probability
        elif category == "goal":
            self.goal_beliefs[belief_name] = new_probability
        elif category == "action":
            self.action_beliefs[belief_name] = new_probability
    
    def get_belief(self, belief_name: str, category: str, default: float = 0.5) -> float:
        """Get current belief probability."""
        if category == "location":
            return self.location_beliefs.get(belief_name, default)
        elif category == "inventory":
            return self.inventory_beliefs.get(belief_name, default)
        elif category == "flag":
            return self.flag_beliefs.get(belief_name, default)
        elif category == "goal":
            return self.goal_beliefs.get(belief_name, default)
        elif category == "action":
            return self.action_beliefs.get(belief_name, default)
        return default
    
    def set_certain(self, belief_name: str, category: str, value: bool):
        """Set a belief to certainty (from direct observation)."""
        self.update_belief(belief_name, category, 1.0 if value else 0.0)
    
    def to_context_string(self) -> str:
        """Format beliefs for LLM context."""
        lines = []
        
        if self.current_location:
            lines.append(f"Location: {self.current_location} (confidence: {self.location_beliefs.get(self.current_location, 0):.0%})")
        
        # High-confidence inventory
        has_items = [item for item, p in self.inventory_beliefs.items() if p > 0.7]
        if has_items:
            lines.append(f"Inventory: {', '.join(has_items)}")
        
        # High-confidence flags
        true_flags = [flag for flag, p in self.flag_beliefs.items() if p > 0.7]
        if true_flags:
            lines.append(f"Known state: {', '.join(true_flags)}")
        
        # Accomplished goals
        done_goals = [goal for goal, p in self.goal_beliefs.items() if p > 0.7]
        if done_goals:
            lines.append(f"Accomplished: {', '.join(done_goals)}")
        
        return "\n".join(lines) if lines else "No confident beliefs yet."
```

### Dynamics Model

```python
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Set
import hashlib

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
    observation_text: str  # For learning/debugging

class DynamicsModel:
    """
    Learned dynamics from direct experience.
    
    In a deterministic game, one observation = certainty.
    """
    
    def __init__(self):
        # (state, action) -> observed outcome
        self.observations: Dict[StateActionKey, ObservedOutcome] = {}
        
        # Track which actions we've tried (regardless of state)
        self.tried_actions: Set[str] = set()
        
        # Track total observations for stats
        self.total_observations: int = 0
    
    def has_observation(self, state_hash: str, action: str) -> bool:
        """Have we tried this action in this state?"""
        key = StateActionKey(state_hash, action)
        return key in self.observations
    
    def get_observation(self, state_hash: str, action: str) -> Optional[ObservedOutcome]:
        """Get observed outcome if we have one."""
        key = StateActionKey(state_hash, action)
        return self.observations.get(key)
    
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
        """
        Get known reward for state-action pair.
        
        Returns None if we haven't observed this pair.
        """
        obs = self.get_observation(state_hash, action)
        return obs.reward if obs else None
    
    def get_stats(self) -> dict:
        return {
            "total_observations": self.total_observations,
            "unique_state_actions": len(self.observations),
            "unique_actions_tried": len(self.tried_actions),
        }
```

---

## The Agent

### Value of Information

For a question $Q$ about action $a'$:

1. **Predict LLM response distribution:**
   $$P(\text{yes}) = \text{TPR} \cdot b + \text{FPR} \cdot (1-b)$$
   where $b = P(\text{true} | B)$ is current belief

2. **Compute posterior beliefs for each answer:**
   $$B_{\text{yes}} = \text{BayesUpdate}(B, \text{yes})$$
   $$B_{\text{no}} = \text{BayesUpdate}(B, \text{no})$$

3. **Compute best action under each posterior:**
   $$a^*_{\text{yes}} = \arg\max_a \mathbb{E}[U(a) | B_{\text{yes}}]$$
   $$a^*_{\text{no}} = \arg\max_a \mathbb{E}[U(a) | B_{\text{no}}]$$

4. **Value of information:**
   $$\text{VOI}(Q) = P(\text{yes}) \cdot \mathbb{E}[U(a^*_{\text{yes}}) | B_{\text{yes}}] + P(\text{no}) \cdot \mathbb{E}[U(a^*_{\text{no}}) | B_{\text{no}}] - \mathbb{E}[U(a^*) | B]$$

This is the expected improvement in action value from asking. It's positive only when the answer might change which action is chosen.

```python
class UnifiedDecisionMaker:
    """
    Unified decision-making over game actions and LLM queries.
    
    Both action types evaluated by expected utility.
    """
    
    def __init__(self, question_cost: float = 0.01):
        """
        Args:
            question_cost: Cost c of asking one question (in reward units).
                          Represents value of computation time.
        """
        self.question_cost = question_cost
    
    def choose(
        self,
        game_actions: List[str],
        possible_questions: List[Tuple[str, str]],  # (action, question)
        beliefs: Dict[str, float],
        sensor: BinarySensor,
        dynamics: 'DynamicsModel',
        state_hash: str,
    ) -> Tuple[str, str]:
        """
        Choose between asking a question or taking a game action.
        
        Returns: (type, value) where type is 'ask' or 'take'
        """
        # Compute EU for all game actions
        game_eus = {}
        for action in game_actions:
            known = dynamics.known_reward(state_hash, action)
            if known is not None:
                game_eus[action] = known
            else:
                belief = beliefs.get(action, 0.5)
                game_eus[action] = belief * 1.0  # EU = P(helps) * reward_if_helps
        
        best_game_action = max(game_actions, key=lambda a: game_eus[a])
        best_game_eu = game_eus[best_game_action]
        
        # Compute EU for all questions (VOI - cost)
        best_question = None
        best_question_eu = float('-inf')
        
        for action, question in possible_questions:
            # Skip if we already know outcome (no uncertainty to resolve)
            if dynamics.has_observation(state_hash, action):
                continue
            
            belief = beliefs.get(action, 0.5)
            voi = self.compute_voi(action, belief, sensor, game_eus)
            question_eu = voi - self.question_cost
            
            if question_eu > best_question_eu:
                best_question_eu = question_eu
                best_question = (action, question)
        
        # Unified decision: take whichever has higher EU
        if best_question is not None and best_question_eu > best_game_eu:
            return ('ask', best_question)
        else:
            return ('take', best_game_action)
    
    def compute_voi(
        self,
        action: str,
        current_belief: float,
        sensor: BinarySensor,
        all_game_eus: Dict[str, float],
    ) -> float:
        """
        Compute value of information for asking about this action.
        
        VOI = E[max EU after asking] - max EU now
        """
        current_best_eu = max(all_game_eus.values())
        
        # P(LLM says yes)
        p_yes = sensor.tpr * current_belief + sensor.fpr * (1 - current_belief)
        p_no = 1 - p_yes
        
        # Posterior beliefs after each answer
        posterior_if_yes = sensor.posterior(current_belief, said_yes=True)
        posterior_if_no = sensor.posterior(current_belief, said_yes=False)
        
        # EU of this action under each posterior
        eu_if_yes = posterior_if_yes * 1.0
        eu_if_no = posterior_if_no * 1.0
        
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
```

### The Agent

```python
from jericho import FrotzEnv
from typing import List, Dict, Tuple, Optional
import hashlib

class BayesianIFAgent:
    """
    Bayesian IF agent with LLM as queryable sensor bank.
    
    Unified decision-making: asking questions and taking game actions
    are both evaluated by expected utility in the same framework.
    
    Core loop:
    1. Generate possible questions and game actions
    2. Compute EU for each (VOI - cost for questions, E[R] for game actions)
    3. Choose argmax
    4. If question: ask, update beliefs, repeat from 1
    5. If game action: take it, observe outcome, learn
    """
    
    def __init__(
        self, 
        llm_client,
        question_cost: float = 0.01,
    ):
        """
        Args:
            llm_client: Client for LLM queries
            question_cost: Cost of asking one question (in reward units).
                          Represents value of computation time.
        """
        self.sensor_bank = LLMSensorBank(llm_client)
        self.dynamics = DynamicsModel()
        self.beliefs = BeliefState()
        
        self.decision_maker = UnifiedDecisionMaker(question_cost=question_cost)
        
        # Tracking
        self.current_state_hash: Optional[str] = None
        self.last_action: Optional[str] = None
        self.last_llm_predictions: Dict[str, bool] = {}
        
        # Stats
        self.total_questions_asked: int = 0
        self.episode_count: int = 0
    
    def get_state_hash(self, env: FrotzEnv) -> str:
        """Get hash of current game state."""
        return str(env.get_world_state_hash())
    
    def choose_action(
        self, 
        env: FrotzEnv, 
        observation: str,
    ) -> Tuple[str, str]:
        """
        Choose action via unified expected utility maximisation.
        
        Returns: (action, explanation)
        """
        self.sensor_bank.clear_cache()
        
        state_hash = self.get_state_hash(env)
        self.current_state_hash = state_hash
        valid_actions = env.get_valid_actions() or ["look"]
        
        context = f"Observation: {observation[:500]}\n\n{self.beliefs.to_context_string()}"
        
        # Build current belief dict for actions
        action_beliefs = {
            a: self.beliefs.get_belief(a, "action", default=0.5)
            for a in valid_actions
        }
        
        sensor = self.sensor_bank.sensors[QuestionType.ACTION_HELPS]
        
        # Unified decision loop: ask or act?
        while True:
            # Generate possible questions
            possible_questions = [
                (a, f"Will the action '{a}' make meaningful progress toward winning the game?")
                for a in valid_actions
                if not self.dynamics.has_observation(state_hash, a)
            ]
            
            # Unified decision
            decision_type, decision_value = self.decision_maker.choose(
                game_actions=valid_actions,
                possible_questions=possible_questions,
                beliefs=action_beliefs,
                sensor=sensor,
                dynamics=self.dynamics,
                state_hash=state_hash,
            )
            
            if decision_type == 'take':
                # Best choice is a game action
                game_action = decision_value
                eu = action_beliefs.get(game_action, 0.5)
                known = self.dynamics.known_reward(state_hash, game_action)
                if known is not None:
                    explanation = f"EU={known:.3f} (known)"
                else:
                    explanation = f"EU={eu:.3f} (belief={action_beliefs.get(game_action, 0.5):.2f})"
                
                self.last_action = game_action
                return game_action, explanation
            
            else:
                # Best choice is to ask a question
                action_to_ask, question = decision_value
                
                answer, _ = self.sensor_bank.ask(
                    QuestionType.ACTION_HELPS, question, context
                )
                
                # Update beliefs via Bayes
                prior = action_beliefs[action_to_ask]
                posterior = sensor.posterior(prior, answer)
                action_beliefs[action_to_ask] = posterior
                self.beliefs.update_belief(action_to_ask, "action", posterior)
                
                # Track for ground truth learning
                self.last_llm_predictions[action_to_ask] = answer
                
                self.total_questions_asked += 1
                
                # Loop continues: re-evaluate whether to ask more or act
    
    def observe_outcome(
        self,
        env: FrotzEnv,
        action: str,
        observation: str,
        reward: float,
    ):
        """
        Learn from outcome.
        
        - Record in dynamics model
        - Update sensor reliability from ground truth
        """
        next_state_hash = self.get_state_hash(env)
        
        # Record dynamics
        if self.current_state_hash:
            self.dynamics.record_observation(
                state_hash=self.current_state_hash,
                action=action,
                next_state_hash=next_state_hash,
                reward=reward,
                observation_text=observation[:200],
            )
        
        # Learn sensor reliability from ground truth
        # Did the action "help"? Define as: reward > 0 OR state changed
        helped = reward > 0 or next_state_hash != self.current_state_hash
        
        # Update sensor for any predictions we made about this action
        if action in self.last_llm_predictions:
            llm_said_helps = self.last_llm_predictions[action]
            self.sensor_bank.update_from_ground_truth(
                QuestionType.ACTION_HELPS,
                said_yes=llm_said_helps,
                actual_truth=helped,
            )
        
        # Clear predictions for next turn
        self.last_llm_predictions = {}
        
        # Update beliefs based on outcome
        # If we got reward, mark as success
        if reward > 0:
            self.beliefs.set_certain(action, "action", True)
    
    def play_episode(
        self,
        env: FrotzEnv,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """Play one episode."""
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        # Reset per-episode state (but keep learned dynamics and sensor reliability)
        self.beliefs = BeliefState()
        self.last_llm_predictions = {}
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {self.episode_count + 1}")
            print(f"{'='*60}")
            print(obs[:400])
        
        while not env.game_over() and steps < max_steps:
            # Choose action
            action, explanation = self.choose_action(env, obs)
            
            if verbose:
                print(f"\n> {action}")
                print(f"  {explanation}")
            
            # Take action
            old_score = env.get_score()
            obs, reward, done, info = env.step(action)
            new_score = env.get_score()
            steps += 1
            
            if verbose and new_score > old_score:
                print(f"  *** SCORE: {old_score} → {new_score} ***")
            
            # Observe outcome
            self.observe_outcome(env, action, obs, reward)
            
            total_reward += reward
        
        self.episode_count += 1
        
        return {
            "episode": self.episode_count,
            "total_reward": total_reward,
            "final_score": env.get_score(),
            "steps": steps,
            "questions_asked": self.total_questions_asked,
            "dynamics_stats": self.dynamics.get_stats(),
            "sensor_stats": self.sensor_bank.get_all_stats(),
        }
    
    def play_multiple_episodes(
        self,
        game_path: str,
        n_episodes: int = 10,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> List[Dict]:
        """Play multiple episodes, learning across them."""
        env = FrotzEnv(game_path)
        results = []
        
        for i in range(n_episodes):
            result = self.play_episode(
                env, max_steps,
                verbose=(verbose and i < 2)
            )
            results.append(result)
            
            # Print summary
            sensor_stats = self.sensor_bank.sensors[QuestionType.ACTION_HELPS].get_stats()
            print(f"Episode {i+1}: score={result['final_score']}, "
                  f"steps={result['steps']}, "
                  f"sensor_reliability={sensor_stats['reliability']:.2f}")
        
        # Final summary
        scores = [r['final_score'] for r in results]
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Scores: {scores}")
        print(f"Mean: {sum(scores)/len(scores):.2f}, Max: {max(scores)}")
        print(f"Total questions asked: {self.total_questions_asked}")
        print(f"Dynamics: {self.dynamics.get_stats()}")
        
        for qt, sensor in self.sensor_bank.sensors.items():
            stats = sensor.get_stats()
            if stats['ground_truths'] > 0:
                print(f"Sensor {qt.value}: TPR={stats['tpr']:.2f}, FPR={stats['fpr']:.2f}, n={stats['ground_truths']}")
        
        return results
```

---

## Ollama Client

```python
import requests
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    timeout: int = 30

class OllamaClient:
    """Client for local Ollama instance."""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
    
    def complete(self, prompt: str) -> str:
        response = requests.post(
            f"{self.config.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "temperature": self.config.temperature,
                "stream": False,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
```

---

## Runner

```python
#!/usr/bin/env python3
"""
Run the Bayesian IF agent.

Usage:
    python runner.py [game_path] [--episodes N] [--verbose]

Prerequisites:
    1. ollama serve
    2. ollama pull llama3.1:8b
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("game", nargs="?", default="games/905.z5")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="llama3.1:8b")
    args = parser.parse_args()
    
    # Check Ollama
    import requests
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except:
        print("ERROR: Ollama not running. Start with: ollama serve")
        sys.exit(1)
    
    from agent import BayesianIFAgent, OllamaClient, OllamaConfig
    
    config = OllamaConfig(model=args.model)
    client = OllamaClient(config)
    agent = BayesianIFAgent(client)
    
    results = agent.play_multiple_episodes(
        game_path=args.game,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
```

---

## Why This Fixes Loops

No loop detection needed. Here's why:

1. Agent tries "look under bed" in state S
2. Gets reward 0, state doesn't change
3. Dynamics model records: (S, "look under bed") → reward=0
4. Next turn, in same state S
5. EU("look under bed") = known_reward = 0
6. EU(any untried action) = P(helps) × 1 + P(doesn't) × 0 = P(helps) > 0 (unless P(helps) = 0)
7. Agent picks untried action

The loop never forms because **repeating a known-futile action has EU=0**, while **untried actions have EU = P(helps) > 0** (assuming non-zero prior).

No hack. Just rational expected utility maximisation.

---

## Why Sensor Reliability Matters

If the LLM always said "yes, every action helps" (high FPR), the agent would learn this:
- FPR increases toward 1.0
- Sensor reliability = TPR - FPR → 0 or negative
- Posteriors barely update from LLM answers
- Agent falls back to prior beliefs + dynamics

If the LLM is actually helpful (high TPR, low FPR):
- Reliability stays high
- LLM answers meaningfully update beliefs
- Agent focuses on promising actions

The agent automatically learns how much to trust the oracle.

---

## Key Properties

1. **Unified decision space**: Asking questions and taking game actions evaluated in same EU framework
2. **Direct experience dominates**: Once observed, no uncertainty
3. **All decisions are EU maximisation**: No hacks, no overrides
4. **Questions asked rationally**: Only when VOI - cost > best game action EU
5. **Sensor reliability is learned**: Agent discovers how much to trust LLM
6. **No loops possible**: Known-futile actions have EU=0, untried have EU=P(helps)>0
7. **Exploration emerges naturally**: Uncertainty gives higher EU than known-bad outcomes

---

## Metrics to Track

- **Score per episode**: Are we improving?
- **Sensor TPR/FPR**: Is the LLM reliable? Are we learning its reliability?
- **Questions per turn**: How many questions before acting?
- **Question cost sensitivity**: How does varying $c$ affect behaviour?
- **Unique state-actions observed**: Coverage of dynamics
- **Decision explanations**: How often "ask" vs "take"? How often "known" vs "belief"?

---

## Success Criteria

1. **No loops**: Agent never repeats futile actions
2. **Sensor learning**: TPR/FPR converge to true reliability
3. **Rational questioning**: Asks when valuable, acts when not
4. **Exploration**: Agent tries diverse actions, doesn't get stuck
5. **Score improvement**: Later episodes better than earlier
6. **Principled**: All behaviour explainable via EU maximisation
