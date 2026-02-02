#!/usr/bin/env python3
"""
Bayesian IF Agent Runner — v4

Runs the Bayesian IF agent on Jericho games.

Usage:
    python runner.py [game_path] [--episodes N] [--max-steps N] [--verbose]
    python runner.py --model llama3.1:8b --question-cost 0.02
"""

import argparse
import os
from typing import Dict, List, Optional, Tuple

from jericho import FrotzEnv

from core import (
    BinarySensor,
    BeliefState,
    DynamicsModel,
    LLMSensorBank,
    QuestionType,
    UnifiedDecisionMaker,
)


def _moment_match_beta(
    alpha: float,
    beta: float,
    sensor: 'BinarySensor',
    said_yes: bool,
) -> Tuple[float, float]:
    """
    Update Beta(alpha, beta) from a sensor observation, moment-matching
    the posterior back to a Beta distribution.

    The likelihood P(yes | theta) = TPR * theta + FPR * (1 - theta) is linear
    in theta, so the exact posterior is not Beta. We compute the posterior mean
    via Bayes' rule and preserve the total count, incrementing by 1 to reflect
    the new evidence.
    """
    prior_mean = alpha / (alpha + beta)
    posterior_mean = sensor.posterior(prior_mean, said_yes)
    new_total = alpha + beta + 1.0
    return (posterior_mean * new_total, (1.0 - posterior_mean) * new_total)


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
        llm_client=None,
        question_cost: float = 0.01,
        action_cost: float = 0.10,
    ):
        self.sensor_bank: Optional[LLMSensorBank] = (
            LLMSensorBank(llm_client) if llm_client is not None else None
        )
        self.dynamics = DynamicsModel()
        self.beliefs = BeliefState()
        self.decision_maker = UnifiedDecisionMaker(
            question_cost=question_cost,
            action_cost=action_cost,
        )

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
        if self.sensor_bank is not None:
            self.sensor_bank.clear_cache()

        state_hash = self.get_state_hash(env)
        self.current_state_hash = state_hash
        valid_actions = env.get_valid_actions() or ["look"]

        context = f"Observation: {observation[:500]}\n\n{self.beliefs.to_context_string()}"

        # Build current belief dict for actions: Beta(alpha, beta) tuples.
        # Default prior: Beta(1/N, 1-1/N) — mean 1/N, total count 1.
        n = len(valid_actions)
        default_prior = (1.0 / n, 1.0 - 1.0 / n)
        action_beliefs: Dict[str, Tuple[float, float]] = {
            a: self.beliefs.get_action_belief(a) or default_prior
            for a in valid_actions
        }

        sensor = (
            self.sensor_bank.sensors[QuestionType.ACTION_HELPS]
            if self.sensor_bank is not None
            else BinarySensor()
        )

        # Defensive bound on questions per turn (VOI > cost should self-terminate,
        # but this guards against pathological edge cases).
        max_questions_per_turn = 10
        questions_this_turn = 0

        # Unified decision loop: ask or act?
        while True:
            # Generate possible questions (only if we have a sensor bank)
            if self.sensor_bank is not None and questions_this_turn < max_questions_per_turn:
                possible_questions = [
                    (a, f"Will the action '{a}' make meaningful progress toward winning the game?")
                    for a in valid_actions
                    if not self.dynamics.has_observation(state_hash, a)
                ]
            else:
                possible_questions = []

            decision_type, decision_value = self.decision_maker.choose(
                game_actions=valid_actions,
                possible_questions=possible_questions,
                beliefs=action_beliefs,
                sensor=sensor,
                dynamics=self.dynamics,
                state_hash=state_hash,
            )

            if decision_type == 'take':
                game_action = decision_value
                known = self.dynamics.known_reward(state_hash, game_action)
                if known is not None:
                    explanation = f"EU={known:.3f} (known)"
                else:
                    alpha, beta = action_beliefs.get(game_action, default_prior)
                    mean = alpha / (alpha + beta)
                    explanation = f"EU={mean:.3f} (belief={mean:.2f})"

                self.last_action = game_action
                return game_action, explanation

            else:
                action_to_ask, question = decision_value

                answer, _ = self.sensor_bank.ask(
                    QuestionType.ACTION_HELPS, question, context
                )

                # Update belief via Bayes' rule, moment-match back to Beta
                alpha, beta = action_beliefs[action_to_ask]
                action_beliefs[action_to_ask] = _moment_match_beta(
                    alpha, beta, sensor, answer
                )
                self.beliefs.set_action_belief(
                    action_to_ask, *action_beliefs[action_to_ask]
                )

                self.last_llm_predictions[action_to_ask] = answer

                self.total_questions_asked += 1
                questions_this_turn += 1

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

        if self.current_state_hash:
            self.dynamics.record_observation(
                state_hash=self.current_state_hash,
                action=action,
                next_state_hash=next_state_hash,
                reward=reward,
                observation_text=observation[:200],
            )

        # Ground truth: did the action "help"?
        # Only reward counts. Not state change. Noisy ground truth poisons learning.
        helped = reward > 0

        # Update sensor for any predictions we made about this action
        if self.sensor_bank is not None and action in self.last_llm_predictions:
            llm_said_helps = self.last_llm_predictions[action]
            self.sensor_bank.update_from_ground_truth(
                QuestionType.ACTION_HELPS,
                said_yes=llm_said_helps,
                actual_truth=helped,
            )

        self.last_llm_predictions = {}

        # Conjugate Beta update from ground truth
        ab = self.beliefs.get_action_belief(action)
        if ab is not None:
            alpha, beta = ab
            if helped:
                self.beliefs.set_action_belief(action, alpha + 1, beta)
            else:
                self.beliefs.set_action_belief(action, alpha, beta + 1)

    def play_episode(
        self,
        env: FrotzEnv,
        max_steps: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """Play one episode."""
        obs, _info = env.reset()
        total_reward = 0
        steps = 0

        # Reset per-episode state (keep learned dynamics and sensor reliability)
        self.beliefs = BeliefState()
        self.last_llm_predictions = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"Episode {self.episode_count + 1}")
            print(f"{'='*60}")
            print(obs[:400])

        while not env.game_over() and steps < max_steps:
            action, explanation = self.choose_action(env, obs)

            if verbose:
                print(f"\n> {action}")
                print(f"  {explanation}")

            old_score = env.get_score()
            obs, reward, _done, _info = env.step(action)
            new_score = env.get_score()
            steps += 1

            if verbose and new_score > old_score:
                print(f"  *** SCORE: {old_score} -> {new_score} ***")

            self.observe_outcome(env, action, obs, reward)

            total_reward += reward

        self.episode_count += 1

        result = {
            "episode": self.episode_count,
            "total_reward": total_reward,
            "final_score": env.get_score(),
            "steps": steps,
            "questions_asked": self.total_questions_asked,
            "dynamics_stats": self.dynamics.get_stats(),
        }
        if self.sensor_bank is not None:
            result["sensor_stats"] = self.sensor_bank.get_all_stats()

        return result

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
                verbose=(verbose and i < 2),
            )
            results.append(result)

            sensor_info = ""
            if self.sensor_bank is not None:
                stats = self.sensor_bank.sensors[QuestionType.ACTION_HELPS].get_stats()
                sensor_info = f", sensor_reliability={stats['reliability']:.2f}"

            print(
                f"Episode {i+1}: score={result['final_score']}, "
                f"steps={result['steps']}{sensor_info}"
            )

        # Final summary
        scores = [r['final_score'] for r in results]
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Scores: {scores}")
        print(f"Mean: {sum(scores)/len(scores):.2f}, Max: {max(scores)}")
        print(f"Total questions asked: {self.total_questions_asked}")
        print(f"Dynamics: {self.dynamics.get_stats()}")

        if self.sensor_bank is not None:
            for qt, sensor in self.sensor_bank.sensors.items():
                stats = sensor.get_stats()
                if stats['ground_truths'] > 0:
                    print(
                        f"Sensor {qt.value}: TPR={stats['tpr']:.2f}, "
                        f"FPR={stats['fpr']:.2f}, n={stats['ground_truths']}"
                    )

        env.close()
        return results


# =============================================================================
# CLI
# =============================================================================

def check_ollama() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Bayesian IF Agent v4")
    parser.add_argument(
        "game", nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "games", "905.z5"),
        help="Path to game file",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--question-cost", type=float, default=0.01)
    parser.add_argument("--action-cost", type=float, default=0.10)
    args = parser.parse_args()

    print("Bayesian IF Agent v4")
    print("=" * 60)

    # Check for Ollama
    llm_client = None
    if check_ollama():
        from ollama_client import OllamaClient, OllamaConfig
        config = OllamaConfig(model=args.model)
        llm_client = OllamaClient(config)
        print(f"LLM: ACTIVE (Ollama detected, model={args.model})")
    else:
        print("LLM: INACTIVE (Ollama not running, using dynamics-only)")

    agent = BayesianIFAgent(
        llm_client=llm_client,
        question_cost=args.question_cost,
        action_cost=args.action_cost,
    )

    agent.play_multiple_episodes(
        game_path=args.game,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
