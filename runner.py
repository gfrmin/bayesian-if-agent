"""
Jericho Game Runner

Connects the Bayesian IF agent to actual games via Jericho.
Tracks performance and provides analysis.

LLM sensor is optional — falls back to uniform priors when Ollama
is unavailable.
"""

from jericho import FrotzEnv
from core import BayesianIFAgent, GameState
from contradiction import detect_contradictions
from typing import List, Dict, Optional, Tuple
import os
import json


def extract_state(env: FrotzEnv) -> GameState:
    """
    Extract game state from Jericho environment using ground truth.
    """
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


class JerichoRunner:
    """
    Runs the Bayesian agent on Jericho games.
    """

    def __init__(self, game_path: str, seed: Optional[int] = None):
        self.game_path = game_path
        self.seed = seed
        self.env = None
        self.agent = None

        # Tracking
        self.episode_history: List[Dict] = []
        self.total_episodes = 0

    def reset(self, agent: Optional[BayesianIFAgent] = None) -> str:
        """
        Reset the game and optionally the agent.

        Returns the initial observation.
        """
        if self.env is not None:
            self.env.close()

        self.env = FrotzEnv(self.game_path, seed=self.seed)
        obs, info = self.env.reset()

        if agent is not None:
            self.agent = agent
        elif self.agent is None:
            self.agent = BayesianIFAgent()

        # Extract ground truth state and give to agent
        state = extract_state(self.env)
        self.agent.observe(state, obs, self.env.get_score())

        self.total_episodes += 1

        return obs

    def step(self, action: Optional[str] = None) -> Tuple[str, float, bool, Dict]:
        """
        Take a step in the game.

        If action is None, the agent chooses.
        """
        if self.env is None:
            raise RuntimeError("Call reset() before step()")

        valid_actions = self.env.get_valid_actions()

        if action is None:
            last_obs = self.agent.history[-1].get("observation", "") if self.agent.history else ""
            action = self.agent.act(valid_actions, observation=last_obs)

        obs, reward, done, info = self.env.step(action)

        # Extract ground truth state and update agent
        state = extract_state(self.env)
        self.agent.observe(state, obs, self.env.get_score())

        info_dict = {
            "action": action,
            "observation": obs,
            "reward": reward,
            "score": self.env.get_score(),
            "done": done,
            "valid_actions": valid_actions[:10],
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
        """
        Play a complete episode.
        """
        obs = self.reset()

        if verbose:
            print("=" * 60)
            print("STARTING NEW EPISODE")
            print("=" * 60)
            print(obs)
            print(f"Score: {self.env.get_score()}")

        walkthrough = None
        walkthrough_idx = 0
        if use_walkthrough:
            walkthrough = self.env.get_walkthrough()
            if verbose:
                print(f"\nWalkthrough ({len(walkthrough)} steps): {walkthrough}")

        episode_log = []
        done = False
        step_count = 0

        while not done and step_count < max_steps:
            if use_walkthrough and walkthrough and walkthrough_idx < len(walkthrough):
                action = walkthrough[walkthrough_idx]
                walkthrough_idx += 1
            else:
                valid_actions = self.env.get_valid_actions()
                last_obs = self.agent.history[-1].get("observation", "") if self.agent.history else ""
                action = self.agent.act(valid_actions, observation=last_obs)

            obs, reward, done, info = self.step(action)
            step_count += 1

            episode_log.append({
                "step": step_count,
                "action": action,
                "reward": reward,
                "score": info["score"],
                "state": str(self.agent.current_state),
            })

            if verbose:
                print(f"\n--- Step {step_count} ---")
                print(f"Action: {action}")
                print(f"Response: {obs[:200]}...")
                print(f"Score: {info['score']} (reward: {reward})")
                print(f"State: {self.agent.current_state}")

                llm_score = self.agent.selector.get_llm_score(action)
                if llm_score != 0.5:  # Only show if sensor is providing non-uniform scores
                    print(f"LLM relevance: {llm_score:.2f}")

            if info.get("game_over") or info.get("victory"):
                done = True

        stats = {
            "total_steps": step_count,
            "final_score": self.env.get_score(),
            "max_score": self.env.get_max_score(),
            "victory": info.get("victory", False),
            "game_over": info.get("game_over", False),
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

        The agent retains learned dynamics across episodes.
        Runs contradiction detection as a diagnostic after each episode.
        """
        all_stats = []
        total_contradictions = 0

        for i in range(n_episodes):
            print(f"\n{'='*40}")
            print(f"Episode {i+1}/{n_episodes}")
            print(f"{'='*40}")

            stats = self.play_episode(
                max_steps=max_steps_per_episode,
                verbose=verbose,
            )
            all_stats.append(stats)

            print(f"Score: {stats['final_score']}/{stats['max_score']} in {stats['total_steps']} steps")
            print(f"Transitions learned: {stats['agent_stats']['transitions_learned']}")

            # Contradiction detection (diagnostic only — no expansion)
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

    def get_learned_dynamics_summary(self) -> str:
        """Get a human-readable summary of what the agent has learned."""
        if self.agent is None:
            return "No agent initialized"

        lines = ["Learned Dynamics:", "=" * 40]

        transitions_by_action: Dict[str, List] = {}
        for state, action, next_state, reward in self.agent.dynamics.observed_transitions:
            if action not in transitions_by_action:
                transitions_by_action[action] = []
            transitions_by_action[action].append((state, next_state, reward))

        for action in sorted(transitions_by_action.keys()):
            lines.append(f"\nAction: '{action}'")
            for state, next_state, reward in transitions_by_action[action][:3]:
                lines.append(f"  loc={state.location} -> loc={next_state.location} (reward: {reward})")

        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    game_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games", "905.z5")

    print("Bayesian IF Agent - Jericho Runner")
    print("=" * 60)

    # Detect LLM sensor availability
    sensor = None
    sensor_model = None

    try:
        import requests
        from ollama_client import OllamaClient, OllamaConfig
        from action_sensor import LLMActionSensor
        from sensor_model import LLMSensorModel

        config = OllamaConfig()
        resp = requests.get(f"{config.base_url}/api/tags", timeout=5)
        resp.raise_for_status()

        # Check if the configured model is available
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        model_available = any(config.model in m for m in models)

        if not model_available:
            print(f"LLM sensor: INACTIVE (model '{config.model}' not found)")
            print(f"  Available models: {models[:5]}")
            print(f"  To install: ollama pull {config.model}")
            raise RuntimeError("model not available")

        client = OllamaClient(config)
        sensor = LLMActionSensor(client)
        sensor_model = LLMSensorModel()

        print(f"LLM sensor: ACTIVE (Ollama detected, model={config.model})")
    except Exception:
        print("LLM sensor: INACTIVE (using uniform priors)")

    # Create agent
    agent = BayesianIFAgent(
        sensor=sensor,
        sensor_model=sensor_model,
        exploration_weight=0.2,
    )

    # Test 1: Play with walkthrough to verify game works
    print("\n\n### TEST 1: Following walkthrough ###")
    runner = JerichoRunner(game_path)
    runner.agent = agent
    stats = runner.play_episode(max_steps=50, verbose=True, use_walkthrough=True)

    # Test 2: Play multiple episodes with learning
    print("\n\n### TEST 2: Learning over multiple episodes ###")
    runner.agent = BayesianIFAgent(
        sensor=sensor,
        sensor_model=sensor_model,
        exploration_weight=0.3,
    )

    summary = runner.play_multiple_episodes(
        n_episodes=5,
        max_steps_per_episode=30,
        verbose=False,
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

    if sensor_model is not None:
        sm_stats = sensor_model.get_statistics()
        print(f"LLM sensor TPR: {sm_stats['true_positive_rate']:.2f}")
        print(f"LLM sensor FPR: {sm_stats['false_positive_rate']:.2f}")

    print("\n" + runner.get_learned_dynamics_summary())
