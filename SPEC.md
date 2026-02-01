# Bayesian IF Agent: Implementation Spec for Claude Code

## Overview

Build an agent that plays interactive fiction games by:
1. Maintaining Bayesian beliefs about game dynamics
2. Expanding its state representation when needed
3. Deciding how much to deliberate before acting
4. Learning across multiple playthroughs

The key insight: **everything is a decision**, including how much to think, what model to use, and whether to expand the model. Simplicity is a utility consideration (cost), not a prior belief.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         JERICHO                                 │
│  Provides: observations, rewards, valid_actions, ground_truth   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STATE PARSER                               │
│                                                                 │
│  Phase 1: Use Jericho ground truth (get_world_state_hash, etc.) │
│  Phase 2: LLM-based parsing (later, once Phase 1 works)         │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DYNAMICS MODEL                              │
│                                                                 │
│  P(next_state, reward | current_state, action)                  │
│  Count-based with pseudocount prior                             │
│  Detects contradictions (same state+action, different outcome)  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATE EXPANDER                               │
│                                                                 │
│  When contradictions detected:                                  │
│  - Find what differs between contradictory cases                │
│  - Propose new state variable                                   │
│  - Evaluate: is expansion worth the cost?                       │
│  - If yes, expand and re-learn                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    METAREASONER                                 │
│                                                                 │
│  Decides what computation to do next:                           │
│  - More inference? (refine beliefs)                             │
│  - Deeper planning? (look further ahead)                        │
│  - Model expansion? (add state variables)                       │
│  - Act now? (stop deliberating)                                 │
│                                                                 │
│  Criterion: V(think) vs V(act)                                  │
│  V(think) = P(improves) × improvement - time_cost               │
│  Stop when V(act) > V(think)                                    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ACTION SELECTOR                               │
│                                                                 │
│  Given beliefs, choose action to maximise expected utility      │
│  Thompson sampling: sample from posterior, pick best action     │
└─────────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. The LLM is a Tool, Not the Agent

The LLM provides services:
- Parsing observations into structured state (Phase 2)
- Proposing state variables when expansion needed
- Generating action candidates if Jericho's valid_actions insufficient

The LLM does NOT:
- Hold beliefs (the Bayesian module does)
- Make decisions (the utility maximiser does)
- Update on observations (Bayes' rule does)

### 2. Bayesian Inference Within Fixed Model

Given a fixed state representation, learn dynamics:

```python
# Transition counts: (state, action) -> {outcome: count}
# Prior: pseudocount (e.g., 0.1) for smoothing
# Update: increment count when transition observed
# Predict: normalise counts to get P(outcome | state, action)
```

### 3. Model Expansion is a Decision

When to expand the state space:

```
Contradiction detected: same (state, action) gave different outcomes
This means state representation is incomplete.

Decision to expand:
  Cost = learning_time + ongoing_complexity_cost
  Benefit = resolved_contradictions × remaining_plays × reward_per_play
  
  if Benefit > Cost: expand
```

### 4. Simplicity is Utility, Not Prior

We don't say "simple models are more likely true."
We say "simple models cost less to use."

```python
U(model) = expected_accuracy × reward - complexity × cost_per_unit_complexity
```

### 5. Deliberation Has Diminishing Returns

More thinking helps less over time:

```python
V(think_more) = P(changes_decision) × improvement - time_cost

P(changes_decision) decreases with each iteration
time_cost accumulates

Eventually: V(act_now) > V(think_more)
```

## Implementation Phases

### Phase 1: Ground Truth State + Basic Learning

Use Jericho's ground truth for state:

```python
from jericho import FrotzEnv

env = FrotzEnv("games/905.z5")

# Ground truth state from Jericho
state_hash = env.get_world_state_hash()  # Unique hash of game state
location = env.get_player_location()      # Current room object
inventory = env.get_inventory()           # Inventory text
score = env.get_score()                   # Current score
valid_actions = env.get_valid_actions()   # Parser-valid actions
```

Build:
- DynamicsModel that learns P(next_hash, reward | current_hash, action)
- ContradictionDetector that flags when same (hash, action) → different next_hash
- Simple ActionSelector using Thompson sampling

Test:
- Play 10+ episodes
- Verify no contradictions (game is deterministic)
- Verify dynamics converge (same transitions seen repeatedly)
- Verify action selection improves with learning

### Phase 2: State Abstraction

The hash is too fine-grained (every state is unique). Abstract to:

```python
@dataclass
class AbstractState:
    location: str       # From env.get_player_location()
    inventory: frozenset  # Items carried
    key_flags: frozenset  # Important world state (doors open, NPCs state, etc.)
```

The question: which flags matter? Start minimal, expand when contradictions found.

Build:
- StateAbstractor that converts Jericho state to AbstractState
- Configurable set of tracked flags
- Logic to detect when abstraction is too coarse (contradictions appear)

### Phase 3: Adaptive State Expansion

When contradictions detected:

```python
def handle_contradiction(state, action, outcomes, history):
    """
    Same (state, action) gave different outcomes.
    Find what's different and propose expansion.
    """
    # Find the observations that led to each outcome
    case_a = find_observation_for_outcome(history, state, action, outcomes[0])
    case_b = find_observation_for_outcome(history, state, action, outcomes[1])
    
    # What's different? Check Jericho's full state
    diff = compute_state_diff(case_a.full_jericho_state, case_b.full_jericho_state)
    
    # Propose a new flag to track
    # e.g., "has_key" or "door_unlocked" or "troll_fed"
    proposal = propose_state_variable(diff)
    
    # Evaluate: is it worth tracking?
    if expansion_is_worthwhile(proposal):
        add_state_variable(proposal)
        reparse_history()  # Re-learn with expanded state
```

### Phase 4: Metareasoning

Add the deliberation budget:

```python
def choose_action(state, budget: float) -> str:
    """
    Choose action within time/computation budget.
    """
    meta_state = quick_eval(state)  # Fast initial assessment
    
    while budget > 0:
        # What's the best use of the next chunk of compute?
        options = [
            ("more_inference", value_of_more_inference(meta_state)),
            ("deeper_planning", value_of_deeper_planning(meta_state)),
            ("check_expansion", value_of_checking_expansion(meta_state)),
            ("act_now", value_of_acting(meta_state)),
        ]
        
        best_option = max(options, key=lambda x: x[1])
        
        if best_option[0] == "act_now":
            break
        
        # Do the computation
        meta_state = execute_meta_action(best_option[0], meta_state)
        budget -= estimated_cost(best_option[0])
    
    return meta_state.best_action
```

### Phase 5: LLM Integration (Optional)

Replace ground-truth parsing with LLM:

```python
def parse_state_with_llm(observation: str, llm) -> AbstractState:
    """
    Use LLM to extract state from observation text.
    """
    prompt = f"""
    Game observation:
    {observation}
    
    Extract:
    1. Location (short identifier)
    2. Items being carried
    3. Important state flags (doors open/closed, NPCs present, etc.)
    
    Return as JSON.
    """
    response = llm.complete(prompt)
    return parse_response(response)
```

Add calibration:
- Compare LLM parsing to ground truth
- Track accuracy per field
- Use accuracy as likelihood in Bayesian update:
  P(true_state | llm_says_X) ∝ P(llm_says_X | true_state) × P(true_state)

## Data Structures

```python
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List, Optional, FrozenSet
from collections import defaultdict

@dataclass(frozen=True)
class AbstractState:
    """Agent's representation of game state."""
    location: str
    inventory: FrozenSet[str]
    flags: FrozenSet[str]

@dataclass
class Transition:
    """A single observed transition."""
    state: AbstractState
    action: str
    next_state: AbstractState
    reward: float
    raw_observation: str  # Keep for re-parsing if model expands

class DynamicsModel:
    """
    Learns P(next_state, reward | state, action).
    Detects contradictions.
    """
    
    def __init__(self, pseudocount: float = 0.1):
        self.pseudocount = pseudocount
        # (state, action) -> {(next_state, reward): count}
        self.counts: Dict[Tuple, Dict[Tuple, float]] = defaultdict(
            lambda: defaultdict(lambda: self.pseudocount)
        )
        self.history: List[Transition] = []
        self.contradictions: List[Tuple] = []
    
    def update(self, transition: Transition):
        """Record a transition, check for contradictions."""
        key = (transition.state, transition.action)
        outcome = (transition.next_state, transition.reward)
        
        # Check for contradiction
        existing = self.counts[key]
        for existing_outcome, count in existing.items():
            if count > self.pseudocount and existing_outcome != outcome:
                self.contradictions.append((key, existing_outcome, outcome))
        
        self.counts[key][outcome] += 1
        self.history.append(transition)
    
    def predict(self, state: AbstractState, action: str) -> Dict[Tuple, float]:
        """P(outcome | state, action)"""
        key = (state, action)
        total = sum(self.counts[key].values())
        return {out: c/total for out, c in self.counts[key].items()}
    
    def sample_outcome(self, state: AbstractState, action: str) -> Optional[Tuple]:
        """Sample from P(outcome | state, action) for Thompson sampling."""
        dist = self.predict(state, action)
        if not dist:
            return None
        outcomes, probs = zip(*dist.items())
        return random.choices(outcomes, probs)[0]
    
    def get_contradictions(self) -> List:
        """Return detected contradictions."""
        return self.contradictions

class StateExpander:
    """Proposes state expansions when contradictions detected."""
    
    def __init__(self, complexity_cost: float = 1.0):
        self.complexity_cost = complexity_cost
        self.current_flags: Set[str] = set()
    
    def propose_expansion(
        self, 
        contradiction, 
        history: List[Transition]
    ) -> Optional[str]:
        """
        Propose a new flag to resolve a contradiction.
        Returns flag name or None.
        """
        (state, action), outcome_a, outcome_b = contradiction
        
        # Find the transitions that led to each outcome
        cases_a = [t for t in history 
                   if t.state == state and t.action == action 
                   and (t.next_state, t.reward) == outcome_a]
        cases_b = [t for t in history
                   if t.state == state and t.action == action
                   and (t.next_state, t.reward) == outcome_b]
        
        if not cases_a or not cases_b:
            return None
        
        # Compare raw observations for hints about what differs
        # In Phase 1, use Jericho ground truth
        # In Phase 5, could use LLM to compare
        
        # For now, return a generic proposal
        return f"unknown_flag_{len(self.current_flags)}"
    
    def evaluate_expansion(
        self, 
        proposal: str, 
        expected_remaining_plays: int
    ) -> bool:
        """Is the expansion worth the cost?"""
        # Cost: complexity
        cost = self.complexity_cost
        
        # Benefit: resolving contradictions helps prediction
        # Rough estimate: 1 unit of value per resolved contradiction per remaining play
        benefit = 0.1 * expected_remaining_plays
        
        return benefit > cost

@dataclass
class MetaState:
    """State of the agent's deliberation."""
    best_action: Optional[str] = None
    best_action_value: float = float('-inf')
    decision_entropy: float = float('inf')
    computation_done: int = 0

class MetaReasoner:
    """Decides how to allocate computation."""
    
    def __init__(self, time_value: float = 0.1):
        self.time_value = time_value
    
    def should_keep_thinking(self, meta_state: MetaState, budget: float) -> bool:
        """Is more deliberation worth it?"""
        if budget <= 0:
            return False
        
        # Value of acting now
        v_act = meta_state.best_action_value
        
        # Value of thinking more
        p_improves = meta_state.decision_entropy * (0.5 ** meta_state.computation_done)
        expected_improvement = meta_state.decision_entropy * 0.1
        v_think = v_act + p_improves * expected_improvement - self.time_value
        
        return v_think > v_act
```

## Testing Strategy

### Test 1: Deterministic dynamics learning

```python
def test_determinism():
    """Game is deterministic. Same state+action should give same outcome."""
    env = FrotzEnv("games/905.z5")
    dynamics = DynamicsModel()
    
    for episode in range(10):
        env.reset()
        while not env.game_over():
            state_hash = env.get_world_state_hash()
            action = random.choice(env.get_valid_actions() or ["look"])
            env.step(action)
            next_hash = env.get_world_state_hash()
            
            # Record with hash as "state" (finest granularity)
            dynamics.update(Transition(
                state=state_hash, action=action,
                next_state=next_hash, reward=env.get_score()
            ))
    
    assert len(dynamics.contradictions) == 0, "Game should be deterministic!"
```

### Test 2: Learning improves action selection

```python
def test_learning_helps():
    """Agent should score better after learning dynamics."""
    
    # Random baseline
    random_scores = [play_episode_randomly() for _ in range(10)]
    
    # Learning agent
    agent = BayesianIFAgent()
    learned_scores = []
    for _ in range(10):
        score = agent.play_episode()  # Agent retains learning
        learned_scores.append(score)
    
    # Later episodes should be better
    assert mean(learned_scores[-5:]) > mean(random_scores)
```

### Test 3: Contradiction detection works

```python
def test_contradiction_detection():
    """Agent detects when abstraction is too coarse."""
    
    # Use deliberately coarse state (location only, no inventory)
    agent = BayesianIFAgent(track_inventory=False)
    
    for _ in range(10):
        agent.play_episode()
    
    # Should have detected contradictions
    # (same location + action gave different outcomes because inventory differed)
    assert len(agent.dynamics.contradictions) > 0
```

### Test 4: Metareasoning stops appropriately

```python
def test_metareasoning():
    """Agent stops deliberating when acting is better."""
    
    agent = BayesianIFAgent()
    
    # Easy decision (one action much better than others)
    deliberation_steps_easy = agent.deliberate(easy_state, budget=1.0)
    
    # Hard decision (actions similar value)
    deliberation_steps_hard = agent.deliberate(hard_state, budget=1.0)
    
    # Should think more for hard decisions
    assert deliberation_steps_hard > deliberation_steps_easy
```

## Suggested File Structure

```
bayesian_if_agent/
├── core/
│   ├── __init__.py
│   ├── state.py          # AbstractState, StateAbstractor
│   ├── dynamics.py       # DynamicsModel, ContradictionDetector
│   ├── expander.py       # StateExpander
│   ├── metareason.py     # MetaReasoner, ComputationBudget
│   └── agent.py          # BayesianIFAgent (ties it together)
├── interface/
│   ├── __init__.py
│   └── jericho_env.py    # Wrapper around Jericho
├── llm/
│   ├── __init__.py
│   ├── parser.py         # LLM-based state parsing (Phase 5)
│   └── proposer.py       # LLM-based expansion proposals
├── tests/
│   ├── test_dynamics.py
│   ├── test_expansion.py
│   ├── test_metareason.py
│   └── test_integration.py
├── experiments/
│   ├── run_learning.py   # Train agent, plot learning curves
│   └── analyze.py        # Analyze what agent learned
├── games/
│   └── 905.z5            # Downloaded game file
├── requirements.txt
└── README.md
```

## Key Metrics to Track

1. **Score per episode** — is the agent improving?
2. **Contradictions detected** — is the state representation sufficient?
3. **Transitions learned** — how much of the state space explored?
4. **Deliberation time** — how much computation per decision?
5. **State variables added** — how complex did the model get?

## What Success Looks Like

Phase 1 success:
- Agent learns dynamics from ground-truth state
- No contradictions with full state hash
- Action selection improves with experience

Phase 2 success:
- Abstract state works for simple games
- Contradictions correctly signal insufficient abstraction

Phase 3 success:
- Agent proposes useful expansions
- Expansions resolve contradictions
- Agent balances complexity vs accuracy

Phase 4 success:
- Agent deliberates more for hard decisions
- Agent stops deliberating when acting is optimal
- No analysis paralysis

Phase 5 success:
- LLM parsing works reliably
- Agent maintains calibrated beliefs about LLM accuracy
