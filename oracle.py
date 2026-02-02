"""
LLM Oracle

Oracle for querying the LLM about anything in the game world.
All responses are observations to condition on, not commands.

Requires: requests (via OllamaClient)
"""

import json
from typing import Dict, List, Any

from ollama_client import OllamaClient


class LLMOracle:
    """
    Oracle for querying the LLM about anything.

    All responses are observations to condition on, not commands.
    """

    def __init__(self, client: OllamaClient):
        self.client = client

    def analyse_situation(self, observation, beliefs, valid_actions):
        """
        Ask LLM for comprehensive situation analysis.

        Returns a SituationUnderstanding (imported lazily to avoid circular deps).
        """
        from core import SituationUnderstanding

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
    "progress_made": true/false,
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

    def analyse_failure(self, observation, beliefs, failed_action, failure_response) -> Dict:
        """Ask LLM why an action failed."""
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

    def detect_progress(self, before_observation, action, after_observation, score_change) -> Dict:
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

    def suggest_state_variables(self, observation, beliefs, contradictions) -> List[str]:
        """Ask LLM what state variables would help."""
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

    def extract_state(self, observation, variables_to_extract) -> Dict[str, Any]:
        """Ask LLM to extract specific state variables from observation."""
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
