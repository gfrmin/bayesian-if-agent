#!/usr/bin/env python3
"""
Setup verification and quick demo script.

Run this after installation to verify everything works.
"""

import sys
import os

def check_dependencies():
    """Check that all required packages are installed."""
    print("Checking dependencies...")
    
    errors = []
    
    # Check jericho
    try:
        import jericho
        print(f"  ✓ jericho {jericho.__version__}")
    except ImportError as e:
        errors.append(f"  ✗ jericho not installed: {e}")
    
    # Check spacy
    try:
        import spacy
        print(f"  ✓ spacy {spacy.__version__}")
        
        # Check for language model
        try:
            nlp = spacy.load("en_core_web_sm")
            print(f"  ✓ en_core_web_sm model loaded")
        except OSError:
            errors.append("  ✗ spacy model not found. Run: uv run python -m spacy download en_core_web_sm")
    except ImportError as e:
        errors.append(f"  ✗ spacy not installed: {e}")
    
    if errors:
        print("\nErrors found:")
        for e in errors:
            print(e)
        return False
    
    print("\nAll dependencies OK!")
    return True


def check_game_file(game_path: str = "games/905.z5") -> bool:
    """Check that a game file exists."""
    print(f"\nChecking for game file: {game_path}")
    
    if os.path.exists(game_path):
        print(f"  ✓ Found {game_path}")
        return True
    else:
        print(f"  ✗ Game file not found: {game_path}")
        print("\n  To download 9:05:")
        print("    mkdir -p games")
        print('    curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5')
        return False


def quick_demo(game_path: str = "games/905.z5"):
    """Run a quick demo of the agent."""
    print("\n" + "=" * 60)
    print("QUICK DEMO")
    print("=" * 60)
    
    from jericho import FrotzEnv
    
    # Test basic game interaction
    print("\nTesting Jericho game interface...")
    env = FrotzEnv(game_path)
    obs, info = env.reset()
    
    print(f"\nGame loaded: {game_path}")
    print(f"Initial observation:\n{obs[:300]}...")
    print(f"\nScore: {env.get_score()}/{env.get_max_score()}")
    print(f"Valid actions: {env.get_valid_actions()[:5]}...")
    
    # Test agent
    print("\n" + "-" * 40)
    print("Testing Bayesian agent...")
    
    from core import BayesianIFAgent
    from runner import EnhancedStateParser
    
    agent = BayesianIFAgent(parser=EnhancedStateParser())
    agent.observe(obs, env.get_score())
    
    print(f"Parsed state: {agent.current_state}")
    
    # Take a few actions
    print("\nTaking a few actions...")
    for i in range(3):
        valid = env.get_valid_actions()
        action = agent.act(valid)
        obs, reward, done, info = env.step(action)
        agent.observe(obs, env.get_score())
        
        print(f"  {i+1}. '{action}' -> score: {env.get_score()}, state: {agent.current_state.location}")
        
        if done:
            break
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Demo complete! The agent is working.")
    print("=" * 60)
    print("\nNext steps:")
    print("  - Run 'uv run python runner.py' for a full training run")
    print("  - See README.md for more details")


def main():
    print("=" * 60)
    print("Bayesian IF Agent - Setup Verification")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Check game file
    game_path = "games/905.z5"
    if not check_game_file(game_path):
        print("\nPlease download a game file and try again.")
        sys.exit(1)
    
    # Run demo
    quick_demo(game_path)


if __name__ == "__main__":
    main()
