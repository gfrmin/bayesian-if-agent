# Adaptive Bayesian IF Agent: Specification v3

## The Core Insight

The LLM is an **oracle** we can query about anything. We've been asking narrow questions ("rate these actions"). We should ask rich questions ("help me understand the situation").

The agent maintains beliefs. The LLM helps refine them. The agent decides.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           GAME (Jericho)                            │
│        Provides: observations, rewards, valid_actions               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AGENT'S BELIEF STATE                           │
│                                                                     │
│  • Current location                                                 │
│  • Inventory                                                        │
│  • Accomplished goals                                               │
│  • Failed actions (with reasons)                                    │
│  • Current goal                                                     │
│  • Blocking conditions                                              │
│  • Tracked state variables                                          │
│  • Dynamics model P(s', r | s, a)                                   │
│  • LLM reliability estimates                                        │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LLM ORACLE (Ollama/Llama3.1)                   │
│                                                                     │
│  We can ask ANYTHING:                                               │
│  • What's the goal?                                                 │
│  • What have I accomplished?                                        │
│  • What's blocking me?                                              │
│  • What action should I take and why?                               │
│  • What state variables matter?                                     │
│  • Did that action make progress?                                   │
│  • Why did that action fail?                                        │
│                                                                     │
│  Input: observation + agent's beliefs + specific questions          │
│  Output: structured understanding (as observations to condition on) │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BAYESIAN DECISION MAKING                       │
│                                                                     │
│  • Update beliefs from LLM outputs (weighted by reliability)        │
│  • Select actions to maximise expected utility                      │
│  • Decide whether to expand state representation                    │
│  • Decide how much to deliberate                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Principle: LLM as Oracle

The LLM is not the agent. The LLM is an oracle that can answer questions about:

| Question Type | Example | Use |
|--------------|---------|-----|
| Goal inference | "What is the objective here?" | Orient planning |
| Progress tracking | "What has been accomplished?" | Avoid repeating |
| Blocker analysis | "Why did 'go east' fail?" | Understand dependencies |
| Action recommendation | "What should I do next?" | Guide exploration |
| State suggestion | "What variables matter?" | Inform representation |
| Progress detection | "Did that action help?" | Learn what works |
| Explanation | "What does this text mean?" | Parse complex situations |

All answers are **observations** — data to condition on, weighted by learned reliability.

---

## Data Structures

### Agent's Belief State

```python
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict

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
            sections.append(f"RECENT ACTIONS: {' → '.join(recent)}")
        
        return '\n'.join(sections)


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
```

### LLM Reliability Tracking

```python
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
```

---

## LLM Oracle Interface

### Setup (Ollama/Llama3.1)

```python
import requests
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class OllamaConfig:
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout: int = 60

class OllamaClient:
    """Client for local Ollama instance."""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.cache: Dict[str, str] = {}
    
    def complete(self, prompt: str, use_cache: bool = True) -> str:
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
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
        result = response.json()["response"]
        
        if use_cache:
            self.cache[cache_key] = result
        return result
    
    def complete_json(self, prompt: str) -> Optional[Dict]:
        full_prompt = prompt + "\n\nRespond with valid JSON only. No other text."
        response = self.complete(full_prompt, use_cache=False)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except json.JSONDecodeError:
            pass
        return None
```

### The Oracle

```python
class LLMOracle:
    """
    Oracle for querying the LLM about anything.
    
    All responses are observations to condition on, not commands.
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    def analyse_situation(
        self,
        observation: str,
        beliefs: AgentBeliefs,
        valid_actions: List[str],
    ) -> SituationUnderstanding:
        """
        Ask LLM for comprehensive situation analysis.
        
        Include everything we know. Get back structured understanding.
        """
        prompt = f"""You are helping an AI agent play a text adventure game.

=== CURRENT OBSERVATION ===
{observation}

=== AGENT'S CURRENT BELIEFS ===
{beliefs.to_prompt_context()}

=== AVAILABLE ACTIONS ===
{json.dumps(valid_actions[:30], indent=2)}

=== YOUR TASK ===
Analyse the situation and provide guidance. Think carefully about:
1. What is the overall goal of this game?
2. What is the immediate next goal?
3. What has already been accomplished?
4. What is currently blocking progress (if anything)?
5. What single action should be taken RIGHT NOW and why?
6. What alternative actions might also help?
7. What state variables are important to track?

Respond with JSON in this exact format:
{{
    "overall_goal": "string - the main objective of the game",
    "immediate_goal": "string - what to focus on right now",
    "accomplished": ["list", "of", "things", "done"],
    "progress_made": true/false - has the agent made progress recently?,
    "blocking_condition": "string or null - what's preventing progress",
    "blocking_reason": "string or null - why this blocks progress",
    "recommended_action": "string - exact action from available list",
    "action_reasoning": "string - why this action helps",
    "alternative_actions": ["other", "good", "actions"],
    "important_state_variables": ["variables", "to", "track"],
    "confidence": 0.0-1.0
}}"""

        result = self.client.complete_json(prompt)
        
        if result:
            return SituationUnderstanding.from_json(result)
        else:
            return SituationUnderstanding()
    
    def analyse_failure(
        self,
        observation: str,
        beliefs: AgentBeliefs,
        failed_action: str,
        failure_response: str,
    ) -> Dict:
        """
        Ask LLM why an action failed.
        """
        prompt = f"""You are helping an AI agent play a text adventure game.

The agent tried an action that didn't work as expected.

=== SITUATION BEFORE ===
{observation}

=== AGENT'S BELIEFS ===
{beliefs.to_prompt_context()}

=== ACTION TRIED ===
{failed_action}

=== GAME'S RESPONSE ===
{failure_response}

=== YOUR TASK ===
Explain why this action failed and what needs to happen first.

Respond with JSON:
{{
    "failure_reason": "string - why the action failed",
    "prerequisite": "string or null - what needs to happen first",
    "suggested_action": "string - what to do instead",
    "lesson": "string - what the agent should learn from this"
}}"""

        return self.client.complete_json(prompt) or {}
    
    def detect_progress(
        self,
        before_observation: str,
        action: str,
        after_observation: str,
        score_change: float,
    ) -> Dict:
        """
        Ask LLM whether an action made progress.
        
        Progress can be score increase OR story advancement.
        """
        prompt = f"""You are helping an AI agent play a text adventure game.

=== BEFORE ACTION ===
{before_observation[:500]}

=== ACTION TAKEN ===
{action}

=== AFTER ACTION ===
{after_observation[:500]}

=== SCORE CHANGE ===
{score_change}

=== YOUR TASK ===
Did this action make meaningful progress toward winning the game?

Consider:
- Did the player reach a new location?
- Did something important change in the game world?
- Is the player closer to their goal?
- Or was this just shuffling items / repeating actions?

Respond with JSON:
{{
    "made_progress": true/false,
    "progress_type": "string - location/item/story/none",
    "explanation": "string - what changed",
    "accomplishment": "string or null - what was accomplished (for tracking)"
}}"""

        return self.client.complete_json(prompt) or {"made_progress": False}
    
    def suggest_state_variables(
        self,
        observation: str,
        beliefs: AgentBeliefs,
        contradictions: List[str],
    ) -> List[str]:
        """
        Ask LLM what state variables would help.
        """
        prompt = f"""You are helping an AI agent play a text adventure game.

The agent is having trouble predicting outcomes. The same action sometimes works and sometimes doesn't.

=== CURRENT OBSERVATION ===
{observation}

=== AGENT'S TRACKED STATE ===
{beliefs.to_prompt_context()}

=== CONTRADICTIONS OBSERVED ===
{json.dumps(contradictions[:5])}

=== YOUR TASK ===
What state variables should the agent track to better predict outcomes?

Think about:
- What hidden conditions affect whether actions work?
- What changes in the game world matter?
- What prerequisites exist for important actions?

Respond with JSON:
{{
    "suggested_variables": ["list", "of", "variable", "names"],
    "reasoning": "string - why these matter"
}}"""

        result = self.client.complete_json(prompt)
        if result and "suggested_variables" in result:
            return result["suggested_variables"]
        return []
    
    def extract_state(
        self,
        observation: str,
        variables_to_extract: List[str],
    ) -> Dict[str, Any]:
        """
        Ask LLM to extract specific state variables from observation.
        """
        prompt = f"""You are helping an AI agent play a text adventure game.

=== CURRENT OBSERVATION ===
{observation}

=== VARIABLES TO EXTRACT ===
{json.dumps(variables_to_extract)}

=== YOUR TASK ===
Extract the value of each variable from the observation.
Use true/false for boolean states, strings for locations/items, null if unknown.

Respond with JSON mapping variable names to values."""

        return self.client.complete_json(prompt) or {}
```

---

## Bayesian Integration

The LLM gives us observations. We update beliefs based on them, weighted by reliability.

```python
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
        """
        Update beliefs based on failure analysis.
        """
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
        """
        Update beliefs based on progress detection.
        """
        if progress.get("made_progress") and progress.get("accomplishment"):
            accomplishment = progress["accomplishment"]
            if accomplishment not in beliefs.accomplished:
                beliefs.accomplished.append(accomplishment)
        
        # Clear blocker if we made progress
        if progress.get("made_progress"):
            beliefs.blocking_condition = None
        
        return beliefs
```

---

## Action Selection

Uses LLM understanding to guide action selection.

```python
import random
from typing import List, Optional, Tuple

class InformedActionSelector:
    """
    Selects actions using LLM understanding + learned dynamics.
    """
    
    def __init__(
        self,
        oracle: LLMOracle,
        reliability: OracleReliability,
        dynamics: 'DynamicsModel',
        exploration_rate: float = 0.2,
    ):
        self.oracle = oracle
        self.reliability = reliability
        self.dynamics = dynamics
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
        
        # Get LLM's situation analysis
        understanding = self.oracle.analyse_situation(
            observation, beliefs, valid_actions
        )
        self.current_understanding = understanding
        
        # Decision: follow recommendation, explore, or use dynamics?
        
        # 1. If LLM has high-confidence recommendation, probably follow it
        if (understanding.recommended_action and 
            understanding.confidence > 0.6 and
            self.reliability.recommendation_accuracy > 0.4):
            
            # Find matching action in valid actions
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
        experienced_actions = [
            a for a in valid_actions 
            if self.dynamics.observation_count(beliefs, a) > 0
        ]
        
        if experienced_actions and random.random() > self.exploration_rate:
            best_action = max(
                experienced_actions,
                key=lambda a: self.dynamics.expected_reward(beliefs, a)
            )
            return best_action, "best learned action"
        
        # 4. Explore: prefer LLM's alternative suggestions
        if understanding.alternative_actions:
            for alt in understanding.alternative_actions:
                for action in valid_actions:
                    if alt.lower() in action.lower():
                        return action, f"exploring LLM alternative"
        
        # 5. Random exploration weighted toward actions LLM hasn't flagged as useless
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
```

---

## Main Agent

```python
from jericho import FrotzEnv
from typing import List, Dict, Optional
import time

class BayesianIFAgent:
    """
    Bayesian IF agent with rich LLM oracle integration.
    
    The LLM helps understand. The agent decides.
    """
    
    def __init__(
        self,
        ollama_config: OllamaConfig = None,
        exploration_rate: float = 0.2,
    ):
        # LLM setup
        self.client = OllamaClient(ollama_config or OllamaConfig())
        self.oracle = LLMOracle(self.client)
        
        # Belief state
        self.beliefs = AgentBeliefs()
        
        # Reliability tracking
        self.reliability = OracleReliability()
        
        # Dynamics model
        self.dynamics = DynamicsModel()
        
        # Action selector
        self.action_selector = InformedActionSelector(
            oracle=self.oracle,
            reliability=self.reliability,
            dynamics=self.dynamics,
            exploration_rate=exploration_rate,
        )
        
        # Belief updater
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
    
    def choose_action(self, env: FrotzEnv, observation: str) -> str:
        """
        Choose action using LLM understanding + Bayesian reasoning.
        """
        # Update beliefs from game state
        self._update_beliefs_from_env(env, observation)
        
        # Get valid actions
        valid_actions = env.get_valid_actions() or ["look"]
        
        # Select action
        action, reason = self.action_selector.select_action(
            observation, self.beliefs, valid_actions
        )
        
        # Record for learning
        self.previous_observation = observation
        self.previous_action = action
        self.previous_score = env.get_score()
        
        # Update action history
        self.beliefs.action_history.append(action)
        if len(self.beliefs.action_history) > 50:
            self.beliefs.action_history = self.beliefs.action_history[-50:]
        
        return action
    
    def observe_outcome(self, env: FrotzEnv, observation: str, reward: float):
        """
        Learn from outcome.
        """
        current_score = env.get_score()
        score_change = current_score - self.previous_score
        
        # Detect progress using LLM
        progress = self.oracle.detect_progress(
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
        if not progress.get("made_progress") and score_change <= 0:
            failure_analysis = self.oracle.analyse_failure(
                self.previous_observation,
                self.beliefs,
                self.previous_action,
                observation,
            )
            self.beliefs = self.belief_updater.update_from_failure_analysis(
                self.beliefs, self.previous_action, failure_analysis
            )
        
        # Update observation history
        self.beliefs.observation_history.append(observation[:200])
        if len(self.beliefs.observation_history) > 20:
            self.beliefs.observation_history = self.beliefs.observation_history[-20:]
    
    def _update_beliefs_from_env(self, env: FrotzEnv, observation: str):
        """Update beliefs from Jericho ground truth."""
        # Location
        location = env.get_player_location()
        if location:
            self.beliefs.location = location.name if hasattr(location, 'name') else str(location.num)
        
        # Inventory
        inventory = env.get_inventory()
        if inventory:
            self.beliefs.inventory = [obj.name for obj in inventory]
        
        # Extract any tracked state variables
        if self.beliefs.tracked_state:
            extracted = self.oracle.extract_state(
                observation,
                list(self.beliefs.tracked_state.keys()),
            )
            self.beliefs.tracked_state.update(extracted)
    
    def play_episode(
        self,
        env: FrotzEnv,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """Play one episode."""
        self.reset_episode()
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
                understanding = self.action_selector.current_understanding
                print(f"\n> {action}")
                if understanding and understanding.action_reasoning:
                    print(f"  Reasoning: {understanding.action_reasoning}")
                if understanding and understanding.blocking_condition:
                    print(f"  Blocker: {understanding.blocking_condition}")
            
            # Take action
            old_score = env.get_score()
            obs, reward, done, info = env.step(action)
            new_score = env.get_score()
            total_reward += reward
            steps += 1
            
            if verbose:
                if new_score > old_score:
                    print(f"  *** SCORE: {old_score} → {new_score} ***")
                print(obs[:300])
            
            # Learn from outcome
            self.observe_outcome(env, obs, reward)
        
        self.episode_count += 1
        
        return {
            "episode": self.episode_count,
            "total_reward": total_reward,
            "final_score": env.get_score(),
            "steps": steps,
            "accomplished": self.beliefs.accomplished,
            "reliability": self.reliability.get_summary(),
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
                verbose=(verbose and i < 2)  # Verbose for first 2 episodes
            )
            results.append(result)
            
            print(f"Episode {i+1}: score={result['final_score']}, "
                  f"steps={result['steps']}, "
                  f"accomplished={len(result['accomplished'])} things")
        
        # Summary
        scores = [r['final_score'] for r in results]
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Mean score: {sum(scores)/len(scores):.2f}")
        print(f"Max score: {max(scores)}")
        print(f"LLM recommendation accuracy: {self.reliability.recommendation_accuracy:.2f}")
        print(f"Overall goal learned: {self.beliefs.overall_goal}")
        
        return results
```

---

## Runner Script

```python
#!/usr/bin/env python3
"""
Run the Bayesian IF agent with rich LLM integration.

Usage:
    python runner.py [game_path] [--episodes N] [--verbose]

Prerequisites:
    1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
    2. Pull model: ollama pull llama3.1:8b
    3. Start Ollama: ollama serve
"""

import argparse
import requests
import sys

def check_ollama():
    """Verify Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    parser = argparse.ArgumentParser(description="Bayesian IF Agent")
    parser.add_argument("game", nargs="?", default="games/905.z5",
                        help="Path to game file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--model", default="llama3.1:8b",
                        help="Ollama model to use")
    args = parser.parse_args()
    
    # Check Ollama
    if not check_ollama():
        print("ERROR: Ollama is not running.")
        print("Start it with: ollama serve")
        print("Then pull the model: ollama pull llama3.1:8b")
        sys.exit(1)
    
    # Import here to avoid issues if dependencies missing
    from agent import BayesianIFAgent, OllamaConfig
    
    # Create agent
    config = OllamaConfig(model=args.model)
    agent = BayesianIFAgent(ollama_config=config)
    
    print(f"Playing {args.game} for {args.episodes} episodes...")
    print(f"Using model: {args.model}")
    print()
    
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
    print(f"\nResults saved to results.json")

if __name__ == "__main__":
    main()
```

---

## Testing

### Test 1: Oracle provides useful understanding

```python
def test_oracle_understanding():
    client = OllamaClient()
    oracle = LLMOracle(client)
    beliefs = AgentBeliefs()
    
    observation = """
    The phone is ringing loudly. You are lying in bed, still groggy.
    The alarm clock shows 9:05 AM. You're going to be late for work!
    """
    
    valid_actions = [
        "answer phone", "sleep", "stand up", "take blanket",
        "put pillow on bed", "look", "inventory"
    ]
    
    understanding = oracle.analyse_situation(observation, beliefs, valid_actions)
    
    print(f"Overall goal: {understanding.overall_goal}")
    print(f"Immediate goal: {understanding.immediate_goal}")
    print(f"Recommended: {understanding.recommended_action}")
    print(f"Reasoning: {understanding.action_reasoning}")
    print(f"Blocker: {understanding.blocking_condition}")
    
    # Should identify "answer phone" or "stand up" as important
    assert understanding.recommended_action is not None
```

### Test 2: Failure analysis works

```python
def test_failure_analysis():
    client = OllamaClient()
    oracle = LLMOracle(client)
    beliefs = AgentBeliefs(location="bedroom")
    
    observation = "You are lying in bed."
    failed_action = "go east"
    failure_response = "You'll have to get up first."
    
    analysis = oracle.analyse_failure(
        observation, beliefs, failed_action, failure_response
    )
    
    print(f"Failure reason: {analysis.get('failure_reason')}")
    print(f"Prerequisite: {analysis.get('prerequisite')}")
    print(f"Suggested: {analysis.get('suggested_action')}")
    
    # Should identify "stand up" as prerequisite
    assert "stand" in analysis.get('prerequisite', '').lower() or \
           "get up" in analysis.get('prerequisite', '').lower()
```

### Test 3: Reliability learning

```python
def test_reliability_learning():
    reliability = OracleReliability()
    
    # Simulate: LLM recommendations help 70% of time
    for _ in range(100):
        helped = random.random() < 0.7
        reliability.update_recommendation(helped)
    
    # Should learn approximately 70% accuracy
    assert 0.6 < reliability.recommendation_accuracy < 0.8
    print(f"Learned accuracy: {reliability.recommendation_accuracy:.2f}")
```

---

## Success Criteria

1. **LLM provides coherent understanding**: Goals, blockers, recommendations make sense
2. **Agent acts on understanding**: Follows recommendations, addresses blockers
3. **Reliability is learned**: Accurate when LLM is helpful, sceptical when not
4. **Progress through dependency chain**: Agent reaches bathroom, living room, car
5. **Score improves over episodes**: Later episodes better than earlier
6. **Principled integration**: LLM outputs are observations, agent decides

---

## Key Differences from v2

| v2 | v3 |
|----|-----|
| LLM rates individual actions | LLM analyses whole situation |
| Single-shot relevance scores | Rich structured understanding |
| No goal tracking | Explicit goal/subgoal tracking |
| No blocker awareness | Blocker detection and addressing |
| No failure analysis | Learn from failures |
| Limited context to LLM | Full belief state in prompts |
| LLM reliability for relevance only | Reliability for multiple query types |

The agent now has a **theory of the situation**, not just action scores.
