"""
LLM Sensor Model

Deprecated: superseded by OracleReliability in core.py. Kept for backward compatibility.

Models the reliability of the LLM sensor using Beta-distributed
true positive and false positive rates, updated from experience.

Stdlib-only — no external imports.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class LLMSensorModel:
    """
    Models the reliability of the LLM sensor.

    The agent learns:
    - P(LLM says relevant | action is actually valuable)  [true positive rate]
    - P(LLM says relevant | action is not valuable)       [false positive rate]

    These are updated from experience via Beta-Bernoulli conjugacy.
    """

    # Beta prior parameters (optimistic: assume LLM is moderately reliable)
    true_positive_alpha: float = 7.0   # ~70% TPR
    true_positive_beta: float = 3.0

    false_positive_alpha: float = 2.0  # ~20% FPR
    false_positive_beta: float = 8.0

    relevance_threshold: float = 0.5
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
            if llm_said_relevant:
                self.true_positive_alpha += 1  # True positive
            else:
                self.true_positive_beta += 1   # False negative
        else:
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
            a = self.true_positive_alpha
            b = self.true_positive_beta
        else:
            a = self.false_positive_alpha
            b = self.false_positive_beta

        # Clamp to avoid log(0) in Beta PDF
        s = max(0.01, min(0.99, llm_score))

        # Beta PDF (unnormalised — fine for likelihood ratios)
        return (s ** (a - 1)) * ((1 - s) ** (b - 1))

    def posterior_valuable(self, llm_score: float, prior_valuable: float = 0.1) -> float:
        """
        P(action valuable | LLM score) via Bayes' rule.

        Args:
            llm_score: The LLM's relevance score
            prior_valuable: Prior probability any action is valuable

        Returns:
            Posterior probability the action is valuable
        """
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
        tp_obs = self.true_positive_alpha + self.true_positive_beta - 10
        fp_obs = self.false_positive_alpha + self.false_positive_beta - 10
        return {
            "true_positive_rate": self.true_positive_rate,
            "false_positive_rate": self.false_positive_rate,
            "tp_observations": tp_obs,
            "fp_observations": fp_obs,
            "total_updates": tp_obs + fp_obs,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import random

    print("LLM Sensor Model - Unit Tests")
    print("=" * 60)

    # Test 1: Basic construction
    model = LLMSensorModel()
    assert 0.6 < model.true_positive_rate < 0.8
    assert 0.1 < model.false_positive_rate < 0.3
    print(f"Initial TPR: {model.true_positive_rate:.2f}")
    print(f"Initial FPR: {model.false_positive_rate:.2f}")

    # Test 2: Learning from experience — LLM is 80% accurate on valuable
    model2 = LLMSensorModel()
    random.seed(42)
    for _ in range(100):
        if random.random() < 0.8:
            model2.update(llm_score=0.8, actual_reward=1.0)  # True positive
        else:
            model2.update(llm_score=0.2, actual_reward=1.0)  # False negative

    assert model2.true_positive_rate > 0.7
    print(f"\nAfter 100 valuable actions (80% accurate LLM):")
    print(f"  Learned TPR: {model2.true_positive_rate:.2f}")

    # Test 3: Posterior updates
    model3 = LLMSensorModel()
    high_score_posterior = model3.posterior_valuable(0.9, prior_valuable=0.1)
    low_score_posterior = model3.posterior_valuable(0.1, prior_valuable=0.1)
    assert high_score_posterior > low_score_posterior
    print(f"\nPosterior P(valuable | LLM=0.9): {high_score_posterior:.3f}")
    print(f"Posterior P(valuable | LLM=0.1): {low_score_posterior:.3f}")

    # Test 4: Statistics
    stats = model2.get_statistics()
    assert "true_positive_rate" in stats
    assert "false_positive_rate" in stats
    print(f"\nStatistics: {stats}")

    print("\nAll sensor model tests passed!")
