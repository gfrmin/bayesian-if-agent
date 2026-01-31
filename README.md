# Bayesian Interactive Fiction Agent

A Bayesian agent that learns to play text adventure games by maintaining beliefs about game dynamics and selecting actions to maximise expected utility.

## Philosophy

This project explores the question: **what should be fixed vs. learned in an intelligent agent?**

**Fixed (the "physics"):**
- Bayesian inference as the update rule
- Expected utility maximisation for action selection
- The game interface: text in, text out, score signal

**Learned within a game:**
- Dynamics: P(next_state, reward | current_state, action)
- State beliefs: P(current_state | observation_history)

**What could evolve across games (future work):**
- Which state variables to track
- Prior hyperparameters
- Exploration/exploitation balance

## Requirements

- Linux (Jericho requires Linux)
- Python 3.9+
- Basic build tools (gcc, make)

## Installation

```bash
# 1. Clone or download this repository
git clone <your-repo-url>
cd bayesian-if-agent

# 2. Install dependencies
uv sync

# 3. Download spacy model (required by Jericho)
uv run python -m spacy download en_core_web_sm

# 4. Download a game file
mkdir -p games
# 9:05 - a short, simple game perfect for testing
curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5

# Or Zork 1 for the classic experience
curl -L "https://www.ifarchive.org/if-archive/games/zcode/zork1.z5" -o games/zork1.z5
```

## Quick Start

```python
from core import BayesianIFAgent
from runner import JerichoRunner, EnhancedStateParser

# Create runner for 9:05
runner = JerichoRunner("games/905.z5")

# Create agent with custom parser
runner.agent = BayesianIFAgent(
    parser=EnhancedStateParser(),
    exploration_bonus=0.2
)

# Play a single episode
stats = runner.play_episode(max_steps=50, verbose=True)

# Or train over multiple episodes
summary = runner.play_multiple_episodes(
    n_episodes=10,
    max_steps_per_episode=30,
    verbose=False
)

print(f"Mean score: {summary['mean_score']}")
print(f"Transitions learned: {summary['final_transitions_learned']}")
```

## Running the Demo

```bash
python runner.py
```

This will:
1. Play through 9:05 following the walkthrough (to verify setup)
2. Train the agent over 5 episodes with learning enabled
3. Print statistics about what the agent learned

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JERICHO INTERFACE                        │
│  env.step(action) → observation, reward, done, info             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STATE PARSER                               │
│  Converts text observations → GameState(location, inventory,    │
│  flags). This is where LLM integration would add value.         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BELIEF STATE                               │
│  P(current_state | observation_history)                         │
│  Particle-based representation of uncertainty                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DYNAMICS MODEL                             │
│  P(next_state, reward | state, action)                          │
│  Count-based with pseudocount prior                             │
│  Updates from every observed transition                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ACTION SELECTOR                            │
│  Thompson sampling or expected utility maximisation             │
│  Balances exploration (uncertain actions) and exploitation      │
└─────────────────────────────────────────────────────────────────┘
```

## Key Files

- `core.py` - Core agent components: GameState, DynamicsModel, BeliefState, ActionSelector, BayesianIFAgent
- `runner.py` - Jericho integration and experiment runner
- `pyproject.toml` - Python dependencies

## Extending the Agent

### Custom State Parser

The default parser uses simple heuristics. You can create a game-specific parser:

```python
from core import StateParser, GameState

class ZorkParser(StateParser):
    def _extract_location(self, obs: str, prev) -> str:
        # Zork-specific location extraction
        if 'west of house' in obs.lower():
            return 'west_of_house'
        # ... etc
        return super()._extract_location(obs, prev)
```

### LLM-Enhanced Parser (Future)

The parser is the natural integration point for an LLM:

```python
class LLMParser(StateParser):
    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client
    
    def parse(self, observation: str, previous_state) -> GameState:
        prompt = f"""
        Given this game observation:
        {observation}
        
        Extract:
        1. Current location (a short identifier)
        2. Items the player is carrying
        3. Important world state flags
        
        Return as JSON.
        """
        response = self.llm.complete(prompt)
        # Parse response into GameState
        ...
```

### Informed Priors

Currently the dynamics model uses uniform priors. An LLM could provide informed priors:

```python
def get_llm_prior(action_type: str) -> float:
    """Ask LLM: how often does this type of action succeed in IF games?"""
    # e.g., "take" actions usually succeed
    # "attack" actions often fail unless you have a weapon
    # etc.
```

## Troubleshooting

### "No module named 'jericho'"
Make sure you've run `uv sync` to install dependencies.

### "FileNotFoundError: games/905.z5"
Download the game file - see Installation step 5.

### Jericho build errors
Jericho requires gcc and make. On Ubuntu/Debian:
```bash
sudo apt-get install build-essential
```

### "spacy model not found"
```bash
uv run python -m spacy download en_core_web_sm
```

## Related Work

- [Jericho](https://github.com/microsoft/jericho) - The IF game interface
- [TextWorld](https://github.com/microsoft/TextWorld) - Procedural IF generation
- [KG-A2C](https://arxiv.org/abs/2002.07626) - Knowledge graph RL for IF
- [TALES](https://microsoft.github.io/tale-suite/) - Text adventure benchmark suite

## Future Directions

1. **LLM Integration**: Use LLMs for state parsing and prior elicitation
2. **Structure Learning**: Learn which state variables matter (not just parameters)
3. **Meta-Learning**: Transfer learned dynamics across similar games
4. **Planning**: Look-ahead search under learned dynamics
5. **Natural Language Beliefs**: Represent beliefs as propositions, not just state vectors

## License

MIT
