"""
State Expander

Proposes, evaluates, and applies new state variables when contradictions
indicate the current representation is too coarse.

All functions are pure — inputs in, outputs out, no mutation.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable, Tuple
from collections import Counter
from core import GameState, Transition, DynamicsModel
from contradiction import Contradiction


@dataclass(frozen=True)
class ExpansionProposal:
    """A proposed new state flag to resolve a contradiction."""
    flag_name: str
    source_contradiction: Contradiction
    estimated_benefit: float
    estimated_cost: float


def _diff_observations(obs_group_a: List[str], obs_group_b: List[str]) -> Optional[str]:
    """
    Compare two groups of raw observation strings to find a distinguishing keyword.

    Looks for words that appear frequently in one group but not the other.
    Returns a candidate flag name, or None if no clear differentiator found.
    """
    def word_counts(texts: List[str]) -> Counter:
        counts: Counter = Counter()
        for text in texts:
            counts.update(text.lower().split())
        return counts

    counts_a = word_counts(obs_group_a)
    counts_b = word_counts(obs_group_b)

    # Words present in one group but absent/rare in the other
    candidates = []
    all_words = set(counts_a.keys()) | set(counts_b.keys())
    for word in all_words:
        freq_a = counts_a.get(word, 0) / max(len(obs_group_a), 1)
        freq_b = counts_b.get(word, 0) / max(len(obs_group_b), 1)
        diff = abs(freq_a - freq_b)
        if diff > 0.5 and len(word) > 2:
            candidates.append((diff, word))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def propose_expansion(
    contradiction: Contradiction,
    history: List[Transition],
) -> Optional[ExpansionProposal]:
    """
    Propose a new state flag to resolve a contradiction.

    Compares raw observations from transitions leading to each contradictory
    outcome to find what differs.
    """
    state, action = contradiction.state, contradiction.action

    obs_a = [
        t.raw_observation for t in history
        if t.state == state and t.action == action
        and (t.next_state, t.reward) == contradiction.outcome_a
        and t.raw_observation
    ]
    obs_b = [
        t.raw_observation for t in history
        if t.state == state and t.action == action
        and (t.next_state, t.reward) == contradiction.outcome_b
        and t.raw_observation
    ]

    if not obs_a or not obs_b:
        return None

    keyword = _diff_observations(obs_a, obs_b)
    flag_name = keyword if keyword else f"flag_{hash(contradiction) % 10000}"

    # Benefit: resolving this contradiction improves prediction for every
    # future visit to this (state, action) pair.
    benefit = float(contradiction.count_a + contradiction.count_b)

    # Cost: one more flag dimension to track
    cost = 1.0

    return ExpansionProposal(
        flag_name=flag_name,
        source_contradiction=contradiction,
        estimated_benefit=benefit,
        estimated_cost=cost,
    )


def evaluate_expansion(
    proposal: ExpansionProposal,
    remaining_episodes: int,
    complexity_cost: float = 1.0,
) -> bool:
    """
    Decide whether an expansion is worth its cost.

    Benefit = estimated_benefit × remaining_episodes × discount
    Cost = complexity_cost
    """
    benefit = proposal.estimated_benefit * remaining_episodes * 0.1
    cost = complexity_cost + proposal.estimated_cost
    return benefit > cost


def apply_expansion(
    flag_name: str,
    history: List[Transition],
    reparse_fn: Callable[[str, Optional[GameState]], GameState],
) -> List[Transition]:
    """
    Re-parse all history with the new flag available.

    reparse_fn(raw_observation, previous_state) -> GameState
    should be a parser that knows about the new flag.

    Returns a new list of Transition objects (no mutation of inputs).
    """
    new_transitions = []
    prev_state: Optional[GameState] = None

    for t in history:
        if t.raw_observation:
            new_state = reparse_fn(t.raw_observation, prev_state)
            # Re-derive the "state" field from the previous transition's new_state
            reparsed_state = prev_state if prev_state is not None else t.state
            new_t = Transition(
                state=reparsed_state,
                action=t.action,
                next_state=new_state,
                reward=t.reward,
                raw_observation=t.raw_observation,
            )
            new_transitions.append(new_t)
            prev_state = new_state
        else:
            new_transitions.append(t)
            prev_state = t.next_state

    return new_transitions


def rebuild_dynamics(
    transitions: List[Transition],
    pseudocount: float = 0.1,
) -> DynamicsModel:
    """
    Reconstruct a DynamicsModel from a list of transitions.

    Returns a fresh model — no mutation of any existing model.
    """
    model = DynamicsModel(prior_pseudocount=pseudocount)
    for t in transitions:
        model.update(t.state, t.action, t.next_state, t.reward, t.raw_observation)
    return model
