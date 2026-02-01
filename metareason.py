"""
Metareasoner

V(think) vs V(act) deliberation budget — decides when to stop
deliberating and act.

All functions are pure (stdlib-only, no mutation).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
from core import BeliefState, DynamicsModel, ActionSelector


@dataclass
class MetaState:
    """Snapshot of the agent's current deliberation state."""
    best_action: Optional[str] = None
    best_action_value: float = float('-inf')
    action_values: Dict[str, float] = field(default_factory=dict)
    decision_entropy: float = float('inf')
    computation_done: int = 0


@dataclass(frozen=True)
class ComputationBudget:
    """How much deliberation is allowed."""
    max_iterations: int = 10
    time_value: float = 0.01  # Cost per deliberation step


def _action_entropy(action_values: Dict[str, float]) -> float:
    """Shannon entropy of softmax over action values (decision uncertainty)."""
    if not action_values:
        return float('inf')

    vals = list(action_values.values())
    max_v = max(vals)
    # Softmax for numerical stability
    exps = [math.exp(v - max_v) for v in vals]
    total = sum(exps)
    probs = [e / total for e in exps]
    return -sum(p * math.log(p + 1e-10) for p in probs)


def value_of_thinking(meta_state: MetaState, time_value: float) -> float:
    """
    Estimated value of one more deliberation step.

    V(think) = P(changes_decision) × expected_improvement - time_cost

    P(changes_decision) decreases with more computation (diminishing returns).
    """
    p_improves = meta_state.decision_entropy * (0.5 ** meta_state.computation_done)
    expected_improvement = meta_state.decision_entropy * 0.1
    return p_improves * expected_improvement - time_value


def should_keep_thinking(meta_state: MetaState, budget: ComputationBudget) -> bool:
    """Should the agent continue deliberating?"""
    if meta_state.computation_done >= budget.max_iterations:
        return False

    v_think = value_of_thinking(meta_state, budget.time_value)
    return v_think > 0


def deliberate(
    belief: BeliefState,
    dynamics: DynamicsModel,
    selector: ActionSelector,
    candidate_actions: List[str],
    budget: ComputationBudget,
) -> Tuple[str, MetaState]:
    """
    Run Thompson sampling iterations until V(act) > V(think) or budget exhausted.

    Each iteration samples from the belief and dynamics posterior, accumulating
    action value estimates. Stops when further deliberation is not worth its cost.

    Returns (chosen_action, final_meta_state).
    """
    if not candidate_actions:
        return "", MetaState()

    # Accumulate value estimates across iterations
    value_sums: Dict[str, float] = {a: 0.0 for a in candidate_actions}
    value_counts: Dict[str, int] = {a: 0 for a in candidate_actions}

    meta = MetaState()

    while True:
        # One round of Thompson sampling: sample state, sample outcome per action
        sampled_state = belief.sample()
        if sampled_state is None:
            # No belief — pick first action
            meta.best_action = candidate_actions[0]
            meta.best_action_value = 0.0
            break

        for action in candidate_actions:
            outcome = dynamics.sample_outcome(sampled_state, action)
            value = outcome[1] if outcome else 0.0
            value_sums[action] += value
            value_counts[action] += 1

        meta.computation_done += 1

        # Update running averages
        meta.action_values = {
            a: value_sums[a] / max(value_counts[a], 1)
            for a in candidate_actions
        }
        meta.best_action = max(meta.action_values, key=meta.action_values.get)
        meta.best_action_value = meta.action_values[meta.best_action]
        meta.decision_entropy = _action_entropy(meta.action_values)

        if not should_keep_thinking(meta, budget):
            break

    return meta.best_action, meta
