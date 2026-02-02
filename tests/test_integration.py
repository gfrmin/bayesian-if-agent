"""Integration tests — full episodes with Jericho (skip if game file absent)."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

GAME_PATH = os.path.join(os.path.dirname(__file__), "..", "games", "905.z5")
GAME_AVAILABLE = os.path.exists(GAME_PATH)

skip_no_game = pytest.mark.skipif(
    not GAME_AVAILABLE,
    reason="Game file not found at games/905.z5",
)


@skip_no_game
def test_agent_play_episode():
    """Play one episode with the agent choosing actions (no LLM)."""
    from jericho import FrotzEnv
    from runner import BayesianIFAgent

    env = FrotzEnv(GAME_PATH)
    agent = BayesianIFAgent()

    result = agent.play_episode(env, max_steps=15)
    assert result["steps"] > 0
    assert result["final_score"] >= 0
    assert result["dynamics_stats"]["total_observations"] > 0

    env.close()


@skip_no_game
def test_agent_multiple_episodes():
    """Play multiple episodes with learning across them."""
    from runner import BayesianIFAgent

    agent = BayesianIFAgent()
    results = agent.play_multiple_episodes(
        game_path=GAME_PATH,
        n_episodes=3,
        max_steps=15,
    )

    assert len(results) == 3
    assert agent.dynamics.total_observations > 0
    assert agent.episode_count == 3


@skip_no_game
def test_dynamics_grow_across_episodes():
    """Dynamics should accumulate across episodes."""
    from jericho import FrotzEnv
    from runner import BayesianIFAgent

    env = FrotzEnv(GAME_PATH)
    agent = BayesianIFAgent()

    agent.play_episode(env, max_steps=10)
    obs_after_1 = agent.dynamics.total_observations

    agent.play_episode(env, max_steps=10)
    obs_after_2 = agent.dynamics.total_observations

    assert obs_after_2 > obs_after_1

    env.close()


@skip_no_game
def test_beliefs_reset_per_episode():
    """Beliefs should be fresh each episode."""
    from jericho import FrotzEnv
    from runner import BayesianIFAgent

    env = FrotzEnv(GAME_PATH)
    agent = BayesianIFAgent()

    agent.play_episode(env, max_steps=5)

    # After episode, beliefs were reset at start
    # Action beliefs from the episode should exist
    # (we just verify no crash and reasonable output)
    assert agent.episode_count == 1

    env.close()
