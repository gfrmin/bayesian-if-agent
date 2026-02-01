"""
Jericho Game Runner

Connects the Bayesian IF agent to actual games via Jericho.
Tracks performance and provides analysis.
"""

from jericho import FrotzEnv
from core import BayesianIFAgent, StateParser, GameState
from typing import List, Dict, Optional, Tuple
import os
import time
import json


class JerichoRunner:
    """
    Runs the Bayesian agent on Jericho games.
    """
    
    def __init__(self, game_path: str, seed: Optional[int] = None):
        """
        Initialize with a game file.
        
        Args:
            game_path: Path to the .z5 game file
            seed: Optional random seed for reproducibility
        """
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
        # Close existing environment if any
        if self.env is not None:
            self.env.close()
        
        # Create new environment
        self.env = FrotzEnv(self.game_path, seed=self.seed)
        obs, info = self.env.reset()
        
        # Create or reset agent
        if agent is not None:
            self.agent = agent
        elif self.agent is None:
            self.agent = BayesianIFAgent()
        
        # Give initial observation to agent
        self.agent.observe(obs, self.env.get_score())
        
        self.total_episodes += 1
        
        return obs
    
    def step(self, action: Optional[str] = None) -> Tuple[str, float, bool, Dict]:
        """
        Take a step in the game.
        
        If action is None, the agent chooses.
        
        Returns:
            observation: The game's response
            reward: Change in score
            done: Whether the game is over
            info: Additional information
        """
        if self.env is None:
            raise RuntimeError("Call reset() before step()")
        
        # Get valid actions from game
        valid_actions = self.env.get_valid_actions()
        
        # Choose action
        if action is None:
            action = self.agent.act(valid_actions)
        
        # Take action in game
        obs, reward, done, info = self.env.step(action)
        
        # Update agent with new observation
        self.agent.observe(obs, self.env.get_score())
        
        # Build info dict
        info_dict = {
            'action': action,
            'observation': obs,
            'reward': reward,
            'score': self.env.get_score(),
            'done': done,
            'valid_actions': valid_actions[:10],  # First 10 for logging
            'game_over': self.env.game_over(),
            'victory': self.env.victory() if hasattr(self.env, 'victory') else False
        }
        
        return obs, reward, done, info_dict
    
    def play_episode(
        self,
        max_steps: int = 100,
        verbose: bool = True,
        use_walkthrough: bool = False
    ) -> Dict:
        """
        Play a complete episode.
        
        Args:
            max_steps: Maximum steps before termination
            verbose: Whether to print progress
            use_walkthrough: Follow the game's walkthrough (for testing)
        
        Returns:
            Episode statistics
        """
        obs = self.reset()
        
        if verbose:
            print("=" * 60)
            print("STARTING NEW EPISODE")
            print("=" * 60)
            print(obs)
            print(f"Score: {self.env.get_score()}")
        
        # Get walkthrough if requested
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
            # Choose action
            if use_walkthrough and walkthrough and walkthrough_idx < len(walkthrough):
                action = walkthrough[walkthrough_idx]
                walkthrough_idx += 1
            else:
                valid_actions = self.env.get_valid_actions()
                action = self.agent.act(valid_actions)
            
            # Take step
            obs, reward, done, info = self.step(action)
            step_count += 1
            
            episode_log.append({
                'step': step_count,
                'action': action,
                'reward': reward,
                'score': info['score'],
                'state': str(self.agent.current_state)
            })
            
            if verbose:
                print(f"\n--- Step {step_count} ---")
                print(f"Action: {action}")
                print(f"Response: {obs[:200]}...")
                print(f"Score: {info['score']} (reward: {reward})")
                print(f"State: {self.agent.current_state}")
            
            # Check for game over
            if info.get('game_over') or info.get('victory'):
                done = True
        
        # Episode statistics
        stats = {
            'total_steps': step_count,
            'final_score': self.env.get_score(),
            'max_score': self.env.get_max_score(),
            'victory': info.get('victory', False),
            'game_over': info.get('game_over', False),
            'agent_stats': self.agent.get_statistics(),
            'log': episode_log
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
        verbose: bool = False
    ) -> Dict:
        """
        Play multiple episodes to train the agent.
        
        The agent retains learned dynamics across episodes.
        """
        all_stats = []
        
        for i in range(n_episodes):
            print(f"\n{'='*40}")
            print(f"Episode {i+1}/{n_episodes}")
            print(f"{'='*40}")
            
            stats = self.play_episode(
                max_steps=max_steps_per_episode,
                verbose=verbose
            )
            all_stats.append(stats)
            
            print(f"Score: {stats['final_score']}/{stats['max_score']} in {stats['total_steps']} steps")
            print(f"Transitions learned: {stats['agent_stats']['transitions_learned']}")
        
        # Summary statistics
        scores = [s['final_score'] for s in all_stats]
        steps = [s['total_steps'] for s in all_stats]
        
        summary = {
            'episodes': n_episodes,
            'mean_score': sum(scores) / len(scores),
            'max_score_achieved': max(scores),
            'mean_steps': sum(steps) / len(steps),
            'victories': sum(1 for s in all_stats if s.get('victory')),
            'final_transitions_learned': self.agent.get_statistics()['transitions_learned'],
            'all_stats': all_stats
        }
        
        return summary
    
    def get_learned_dynamics_summary(self) -> str:
        """Get a human-readable summary of what the agent has learned."""
        if self.agent is None:
            return "No agent initialized"
        
        lines = ["Learned Dynamics:", "=" * 40]
        
        # Group transitions by action
        transitions_by_action: Dict[str, List] = {}
        for state, action, next_state, reward in self.agent.dynamics.observed_transitions:
            if action not in transitions_by_action:
                transitions_by_action[action] = []
            transitions_by_action[action].append((state, next_state, reward))
        
        for action in sorted(transitions_by_action.keys()):
            lines.append(f"\nAction: '{action}'")
            for state, next_state, reward in transitions_by_action[action][:3]:  # Show first 3
                lines.append(f"  {state.location} -> {next_state.location} (reward: {reward})")
        
        return "\n".join(lines)


class EnhancedStateParser(StateParser):
    """
    Enhanced state parser that uses game-specific knowledge.
    
    For 9:05, we know the specific locations and objects.
    """
    
    def __init__(self):
        super().__init__()
        
        # 9:05 specific locations
        self.known_locations = [
            'bedroom', 'bathroom', 'living room', 'outside', 'car'
        ]
        
        # 9:05 specific items
        self.known_items = [
            'telephone', 'phone', 'wallet', 'keys', 'watch', 
            'clothes', 'clothing', 'dresser'
        ]
    
    def _extract_location(self, obs: str, prev: Optional[GameState]) -> str:
        """Extract location with 9:05 knowledge."""
        # Check for explicit location markers
        if 'bedroom' in obs.lower():
            if 'in bed' in obs.lower():
                return 'bedroom_bed'
            return 'bedroom'
        if 'bathroom' in obs.lower():
            return 'bathroom'
        if 'living room' in obs.lower():
            return 'living_room'
        if 'outside' in obs.lower() or 'driveway' in obs.lower():
            return 'outside'
        if 'car' in obs.lower() and ('in the' in obs.lower() or 'enter' in obs.lower()):
            return 'car'
        
        return prev.location if prev else "unknown"
    
    def _extract_inventory(self, obs: str, prev: Optional[GameState]) -> frozenset:
        """Extract inventory with 9:05 knowledge."""
        inventory = set(prev.inventory) if prev else set()
        obs_lower = obs.lower()
        
        # Taking items
        for item in self.known_items:
            if f'take {item}' in obs_lower or f'taken' in obs_lower:
                if item in obs_lower:
                    inventory.add(item)
            if f'get {item}' in obs_lower:
                inventory.add(item)
        
        # Wearing items
        if 'wear' in obs_lower and 'clothes' in obs_lower:
            inventory.add('clothes_worn')
        if 'wear' in obs_lower and 'watch' in obs_lower:
            inventory.add('watch_worn')
        
        # Dropping items
        if 'drop' in obs_lower or 'remove' in obs_lower:
            for item in list(inventory):
                if item in obs_lower:
                    inventory.discard(item)
        
        return frozenset(inventory)
    
    def _extract_flags(self, obs: str, prev: Optional[GameState]) -> frozenset:
        """Extract flags with 9:05 knowledge."""
        flags = set(prev.flags) if prev else set()
        obs_lower = obs.lower()
        
        # Phone answered
        if 'answer' in obs_lower and 'phone' in obs_lower:
            flags.add('phone_answered')
        
        # Standing up
        if 'stand up' in obs_lower or 'get up' in obs_lower or 'out of bed' in obs_lower:
            flags.add('standing')
        
        # Showered
        if 'shower' in obs_lower:
            flags.add('showered')
        
        # Dressed
        if 'wearing' in obs_lower and 'clothes' in obs_lower:
            flags.add('dressed')
        
        # Door open
        if 'open' in obs_lower and 'door' in obs_lower:
            flags.add('front_door_open')
        
        # Car unlocked
        if 'unlock' in obs_lower and 'car' in obs_lower:
            flags.add('car_unlocked')
        
        return frozenset(flags)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Default game path
    game_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "games", "905.z5")
    
    print("Bayesian IF Agent - Jericho Runner")
    print("=" * 60)
    
    # Create runner with enhanced parser for 9:05
    runner = JerichoRunner(game_path)
    runner.agent = BayesianIFAgent(
        parser=EnhancedStateParser(),
        exploration_bonus=0.2  # Encourage exploration
    )
    
    # Test 1: Play with walkthrough to verify game works
    print("\n\n### TEST 1: Following walkthrough ###")
    stats = runner.play_episode(max_steps=50, verbose=True, use_walkthrough=True)
    
    # Test 2: Play multiple episodes with learning
    print("\n\n### TEST 2: Learning over multiple episodes ###")
    runner.agent = BayesianIFAgent(
        parser=EnhancedStateParser(),
        exploration_bonus=0.3
    )
    
    summary = runner.play_multiple_episodes(
        n_episodes=5,
        max_steps_per_episode=30,
        verbose=False
    )
    
    print("\n" + "=" * 60)
    print("LEARNING SUMMARY")
    print("=" * 60)
    print(f"Episodes played: {summary['episodes']}")
    print(f"Mean score: {summary['mean_score']:.2f}")
    print(f"Max score achieved: {summary['max_score_achieved']}")
    print(f"Mean steps: {summary['mean_steps']:.1f}")
    print(f"Total transitions learned: {summary['final_transitions_learned']}")
    
    print("\n" + runner.get_learned_dynamics_summary())
