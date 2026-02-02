"""
Action Sensors

LLMActionSensor queries Ollama for action relevance scores.
UniformActionSensor is a fallback that returns 0.5 for all actions.

Both implement the same interface:
    get_relevance_scores(observation, valid_actions) -> Dict[str, float]
"""

import json
from typing import Dict, List

from ollama_client import OllamaClient


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
        valid_actions: List[str],
    ) -> Dict[str, float]:
        """
        Query LLM for action relevance scores.

        Returns: {action: relevance} where relevance in [0, 1]
        """
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

        try:
            result = self.client.complete_json(prompt)
        except Exception:
            return {a: 0.5 for a in valid_actions}

        if result is None:
            return {a: 0.5 for a in valid_actions}

        scores = {}
        for action in valid_actions:
            if action in result:
                scores[action] = max(0.0, min(1.0, float(result[action])))
            else:
                scores[action] = 0.3
        return scores


class UniformActionSensor:
    """Fallback sensor that returns 0.5 for all actions (no LLM needed)."""

    def get_relevance_scores(
        self,
        observation: str,
        valid_actions: List[str],
    ) -> Dict[str, float]:
        return {a: 0.5 for a in valid_actions}
