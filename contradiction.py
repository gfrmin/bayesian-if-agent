"""
Contradiction Detection

Finds (state, action) pairs that produce multiple distinct outcomes,
signaling that the state representation is too coarse.

Pure functions — no mutation, no side effects.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
from core import GameState, Transition, DynamicsModel


@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction: same (state, action) yielded different outcomes."""
    state: GameState
    action: str
    outcome_a: Tuple[GameState, float]  # (next_state, reward)
    outcome_b: Tuple[GameState, float]
    count_a: int
    count_b: int


def detect_contradictions(
    dynamics: DynamicsModel,
    threshold: int = 1
) -> List[Contradiction]:
    """
    Find (state, action) pairs with multiple distinct observed outcomes.

    A contradiction exists when the same (state, action) produced at least
    two different (next_state, reward) outcomes, each observed at least
    `threshold` times.

    This is a batch query over the dynamics model's history — intended to
    run end-of-episode, not inside update().
    """
    # Group transitions by (state, action)
    groups: Dict[Tuple[GameState, str], Dict[Tuple[GameState, float], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for t in dynamics.history:
        groups[(t.state, t.action)][(t.next_state, t.reward)] += 1

    contradictions = []
    for (state, action), outcome_counts in groups.items():
        # Filter to outcomes meeting the threshold
        significant = [
            (outcome, count)
            for outcome, count in outcome_counts.items()
            if count >= threshold
        ]
        if len(significant) < 2:
            continue

        # Report pairwise contradictions (sorted for determinism)
        significant.sort(key=lambda x: (-x[1], str(x[0])))
        for i in range(len(significant)):
            for j in range(i + 1, len(significant)):
                outcome_a, count_a = significant[i]
                outcome_b, count_b = significant[j]
                contradictions.append(Contradiction(
                    state=state,
                    action=action,
                    outcome_a=outcome_a,
                    outcome_b=outcome_b,
                    count_a=count_a,
                    count_b=count_b,
                ))

    return contradictions
