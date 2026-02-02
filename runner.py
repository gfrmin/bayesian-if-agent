#!/usr/bin/env python3
"""
Jericho Game Runner — v3

Connects the Bayesian IF agent (with LLM oracle) to actual games via Jericho.
Detects Ollama availability at startup; falls back to dynamics-only when absent.

Usage:
    python runner.py [game_path] [--episodes N] [--max-steps N] [--verbose] [--model MODEL]
"""

from jericho import FrotzEnv
from core import BayesianIFAgent, GameState
from contradiction import detect_contradictions
from typing import List, Dict, Optional, Tuple
import os
import argparse


def extract_state(env: FrotzEnv) -> GameState:
    """Extract game state from Jericho environment using ground truth."""
    loc = env.get_player_location()
    location_id = loc.num if loc else 0

    inv = env.get_inventory()
    inventory = frozenset(obj.name for obj in inv) if inv else frozenset()

    world_hash = str(env.get_world_state_hash())

    return GameState(
        location=location_id,
        inventory=inventory,
        world_hash=world_hash,
    )


def extract_beliefs_data(env: FrotzEnv) -> dict:
    """Extract data for updating agent beliefs from Jericho ground truth."""
    loc = env.get_player_location()
    location_name = loc.name if loc and hasattr(loc, 'name') else str(loc.num if loc else 0)
    location_id = loc.num if loc else 0

    inv = env.get_inventory()
    inventory_list = [obj.name for obj in inv] if inv else []

    world_hash = str(env.get_world_state_hash())

    return {
        "location": location_name,
        "location_id": location_id,
        "inventory": inventory_list,
        "world_hash": world_hash,
    }


def check_ollama() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


class JerichoRunner:
    """
    Runs the Bayesian agent on Jericho games.

    Detects oracle availability at startup and injects it into the agent.
    """

    def __init__(self, game_path: str, seed: Optional[int] = None, model: str = "llama3.1:8b"):
        self.game_path = game_path
        self.seed = seed
        self.model = model
        self.env = None
        self.agent = None

        # Tracking
        self.episode_history: List[Dict] = []
        self.total_episodes = 0

    def _create_oracle(self):
        """Try to create an LLM oracle, return None if unavailable."""
        try:
            import requests
            from ollama_client import OllamaClient, OllamaConfig
            from oracle import LLMOracle

            config = OllamaConfig(model=self.model)
            resp = requests.get(f"{config.base_url}/api/tags", timeout=5)
            resp.raise_for_status()

            models = [m.get("name", "") for m in resp.json().get("models", [])]
            model_available = any(config.model in m for m in models)

            if not model_available:
                print(f"Oracle: INACTIVE (model '{config.model}' not found)")
                print(f"  Available models: {models[:5]}")
                print(f"  To install: ollama pull {config.model}")
                return None

            client = OllamaClient(config)
            oracle = LLMOracle(client)
            print(f"Oracle: ACTIVE (Ollama detected, model={config.model})")
            return oracle

        except Exception:
            print("Oracle: INACTIVE (Ollama not running, using dynamics-only)")
            return None

    def reset(self, agent: Optional[BayesianIFAgent] = None) -> str:
        """Reset the game and optionally the agent. Returns the initial observation."""
        if self.env is not None:
            self.env.close()

        self.env = FrotzEnv(self.game_path, seed=self.seed)
        obs, info = self.env.reset()

        if agent is not None:
            self.agent = agent
        elif self.agent is None:
            oracle = self._create_oracle()
            self.agent = BayesianIFAgent(oracle=oracle, exploration_rate=0.2)

        # Update agent beliefs from ground truth
        data = extract_beliefs_data(self.env)
        self.agent.update_beliefs_from_ground_truth(
            data["location"], data["location_id"],
            data["inventory"], data["world_hash"],
        )

        self.total_episodes += 1
        return obs

    def step(self, action: Optional[str] = None) -> Tuple[str, float, bool, Dict]:
        """Take a step in the game. If action is None, the agent chooses."""
        if self.env is None:
            raise RuntimeError("Call reset() before step()")

        valid_actions = self.env.get_valid_actions()

        if action is None:
            last_obs = ""
            if self.agent.beliefs.observation_history:
                last_obs = self.agent.beliefs.observation_history[-1]
            action = self.agent.choose_action(last_obs, valid_actions or ["look"])

        old_score = self.env.get_score()
        obs, reward, done, info = self.env.step(action)
        new_score = self.env.get_score()

        # Update agent beliefs from ground truth
        data = extract_beliefs_data(self.env)
        self.agent.update_beliefs_from_ground_truth(
            data["location"], data["location_id"],
            data["inventory"], data["world_hash"],
        )

        # Let agent learn from outcome
        self.agent.observe_outcome(obs, reward, new_score)

        info_dict = {
            "action": action,
            "observation": obs,
            "reward": reward,
            "score": new_score,
            "done": done,
            "valid_actions": valid_actions[:10] if valid_actions else [],
            "game_over": self.env.game_over(),
            "victory": self.env.victory() if hasattr(self.env, "victory") else False,
        }

        return obs, reward, done, info_dict

    def play_episode(
        self,
        max_steps: int = 100,
        verbose: bool = True,
        use_walkthrough: bool = False,
    ) -> Dict:
        """Play a complete episode."""
        obs = self.reset()

        if verbose:
            print("=" * 60)
            print("STARTING NEW EPISODE")
            print("=" * 60)
            print(obs)
            print(f"Score: {self.env.get_score()}")
            print(f"Beliefs: {self.agent.beliefs.to_prompt_context()}")

        walkthrough = None
        walkthrough_idx = 0
        if use_walkthrough:
            walkthrough = self.env.get_walkthrough()
            if verbose:
                print(f"\nWalkthrough ({len(walkthrough)} steps): {walkthrough}")

        episode_log = []
        done = False
        step_count = 0

        # Store initial observation for the agent
        self.agent.beliefs.observation_history.append(obs[:200])

        while not done and step_count < max_steps:
            if use_walkthrough and walkthrough and walkthrough_idx < len(walkthrough):
                action = walkthrough[walkthrough_idx]
                walkthrough_idx += 1
                # Still let agent know about the action for learning
                self.agent.previous_observation = obs
                self.agent.previous_action = action
                self.agent.previous_score = self.env.get_score()
            else:
                action = self.agent.choose_action(obs, self.env.get_valid_actions() or ["look"])

            obs, reward, done, info = self.step(action)
            step_count += 1

            understanding = self.agent.action_selector.current_understanding

            episode_log.append({
                "step": step_count,
                "action": action,
                "reward": reward,
                "score": info["score"],
                "reasoning": understanding.action_reasoning if understanding else None,
                "blocker": understanding.blocking_condition if understanding else None,
            })

            if verbose:
                print(f"\n--- Step {step_count} ---")
                print(f"Action: {action}")
                print(f"Response: {obs[:200]}...")
                print(f"Score: {info['score']} (reward: {reward})")
                print(f"Location: {self.agent.beliefs.location}")

                if understanding:
                    if understanding.action_reasoning:
                        print(f"Reasoning: {understanding.action_reasoning}")
                    if understanding.blocking_condition:
                        print(f"Blocker: {understanding.blocking_condition}")
                    if understanding.immediate_goal:
                        print(f"Goal: {understanding.immediate_goal}")

            if info.get("game_over") or info.get("victory"):
                done = True

        self.agent.episode_count += 1

        stats = {
            "episode": self.agent.episode_count,
            "total_steps": step_count,
            "final_score": self.env.get_score(),
            "max_score": self.env.get_max_score(),
            "victory": info.get("victory", False) if step_count > 0 else False,
            "game_over": info.get("game_over", False) if step_count > 0 else False,
            "agent_stats": self.agent.get_statistics(),
            "log": episode_log,
        }

        self.episode_history.append(stats)

        if verbose:
            print("\n" + "=" * 60)
            print("EPISODE COMPLETE")
            print(f"Final score: {stats['final_score']}/{stats['max_score']}")
            print(f"Steps: {stats['total_steps']}")
            print(f"Victory: {stats['victory']}")
            print(f"Goal learned: {self.agent.beliefs.overall_goal}")
            print(f"Accomplished: {self.agent.beliefs.accomplished}")
            print(f"Reliability: {self.agent.reliability.get_summary()}")
            print("=" * 60)

        return stats

    def play_multiple_episodes(
        self,
        n_episodes: int,
        max_steps_per_episode: int = 100,
        verbose: bool = False,
    ) -> Dict:
        """
        Play multiple episodes to train the agent.

        The agent retains learned dynamics and beliefs across episodes.
        Runs contradiction detection as a diagnostic after each episode.
        """
        all_stats = []
        total_contradictions = 0

        for i in range(n_episodes):
            print(f"\n{'='*40}")
            print(f"Episode {i+1}/{n_episodes}")
            print(f"{'='*40}")

            # Reset episode but keep learned knowledge
            self.agent.reset_episode()

            stats = self.play_episode(
                max_steps=max_steps_per_episode,
                verbose=verbose,
            )
            all_stats.append(stats)

            print(f"Score: {stats['final_score']}/{stats['max_score']} in {stats['total_steps']} steps")
            print(f"Transitions learned: {stats['agent_stats']['transitions_learned']}")
            print(f"Goal: {stats['agent_stats']['overall_goal']}")
            print(f"Accomplished: {len(stats['agent_stats']['accomplished'])} things")

            # Reliability
            rel = stats['agent_stats']['reliability']
            print(f"Recommendation accuracy: {rel['recommendation_accuracy']:.2f} "
                  f"(n={rel['total_recommendations']})")

            # Rewarded actions
            rewarded = [e for e in stats["log"] if e["reward"] > 0]
            if rewarded:
                print(f"Rewarded actions: {', '.join(e['action'] for e in rewarded)}")

            # Contradiction detection (diagnostic only)
            if self.agent is not None:
                contradictions = detect_contradictions(self.agent.dynamics)
                episode_contradictions = len(contradictions)
                total_contradictions += episode_contradictions

                if contradictions:
                    print(f"Contradictions detected: {episode_contradictions}")

        scores = [s["final_score"] for s in all_stats]
        steps = [s["total_steps"] for s in all_stats]

        summary = {
            "episodes": n_episodes,
            "mean_score": sum(scores) / len(scores),
            "max_score_achieved": max(scores),
            "mean_steps": sum(steps) / len(steps),
            "victories": sum(1 for s in all_stats if s.get("victory")),
            "final_transitions_learned": self.agent.get_statistics()["transitions_learned"],
            "total_contradictions_detected": total_contradictions,
            "all_stats": all_stats,
        }

        return summary


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian IF Agent v3")
    parser.add_argument("game", nargs="?",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "games", "905.z5"),
                        help="Path to game file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--model", default="llama3.1:8b",
                        help="Ollama model to use")
    args = parser.parse_args()

    print("Bayesian IF Agent v3 - Jericho Runner")
    print("=" * 60)

    runner = JerichoRunner(args.game, model=args.model)

    # Test 1: Play with walkthrough to verify game works
    print("\n\n### TEST 1: Following walkthrough ###")
    stats = runner.play_episode(max_steps=50, verbose=True, use_walkthrough=True)

    # Test 2: Play multiple episodes with learning
    print("\n\n### TEST 2: Learning over multiple episodes ###")
    # Create fresh agent for learning run
    oracle = runner._create_oracle()
    runner.agent = BayesianIFAgent(oracle=oracle, exploration_rate=0.3)

    summary = runner.play_multiple_episodes(
        n_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("LEARNING SUMMARY")
    print("=" * 60)
    print(f"Episodes played: {summary['episodes']}")
    print(f"Mean score: {summary['mean_score']:.2f}")
    print(f"Max score achieved: {summary['max_score_achieved']}")
    print(f"Mean steps: {summary['mean_steps']:.1f}")
    print(f"Total transitions learned: {summary['final_transitions_learned']}")
    print(f"Contradictions detected: {summary.get('total_contradictions_detected', 0)}")
    print(f"Goal learned: {runner.agent.beliefs.overall_goal}")
    print(f"Reliability: {runner.agent.reliability.get_summary()}")
