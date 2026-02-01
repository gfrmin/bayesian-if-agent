"""Integration tests — full episodes with Jericho (skip if game file absent)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

GAME_PATH = os.path.join(os.path.dirname(__file__), "..", "games", "905.z5")
GAME_AVAILABLE = os.path.exists(GAME_PATH)

skip_no_game = pytest.mark.skipif(
    not GAME_AVAILABLE,
    reason="Game file not found at games/905.z5"
)


@skip_no_game
def test_runner_walkthrough():
    """Play one episode using the game's walkthrough."""
    from runner import JerichoRunner, EnhancedStateParser
    from core import BayesianIFAgent

    runner = JerichoRunner(GAME_PATH)
    runner.agent = BayesianIFAgent(parser=EnhancedStateParser())

    stats = runner.play_episode(max_steps=50, verbose=False, use_walkthrough=True)
    assert stats['total_steps'] > 0
    assert stats['final_score'] >= 0

    runner.env.close()


@skip_no_game
def test_runner_agent_play():
    """Play one episode with the agent choosing actions."""
    from runner import JerichoRunner, EnhancedStateParser
    from core import BayesianIFAgent

    runner = JerichoRunner(GAME_PATH)
    runner.agent = BayesianIFAgent(
        parser=EnhancedStateParser(),
        exploration_bonus=0.3,
    )

    stats = runner.play_episode(max_steps=15, verbose=False)
    assert stats['total_steps'] > 0
    assert stats['agent_stats']['dynamics_history_size'] >= 0

    runner.env.close()


@skip_no_game
def test_runner_multiple_episodes_with_expansion():
    """Play multiple episodes with contradiction detection and expansion."""
    from runner import JerichoRunner, EnhancedStateParser
    from core import BayesianIFAgent

    runner = JerichoRunner(GAME_PATH)
    runner.agent = BayesianIFAgent(
        parser=EnhancedStateParser(),
        exploration_bonus=0.3,
    )

    summary = runner.play_multiple_episodes(
        n_episodes=3,
        max_steps_per_episode=15,
        verbose=False,
        enable_expansion=True,
    )

    assert summary['episodes'] == 3
    assert 'total_contradictions_detected' in summary
    assert 'variables_added' in summary
    assert summary['final_transitions_learned'] >= 0

    runner.env.close()


@skip_no_game
def test_dynamics_history_grows_across_episodes():
    """Dynamics history should accumulate across episodes."""
    from runner import JerichoRunner, EnhancedStateParser
    from core import BayesianIFAgent

    runner = JerichoRunner(GAME_PATH)
    runner.agent = BayesianIFAgent(
        parser=EnhancedStateParser(),
        exploration_bonus=0.3,
    )

    runner.play_episode(max_steps=10, verbose=False)
    history_after_1 = len(runner.agent.dynamics.history)

    runner.play_episode(max_steps=10, verbose=False)
    history_after_2 = len(runner.agent.dynamics.history)

    assert history_after_2 > history_after_1

    runner.env.close()
