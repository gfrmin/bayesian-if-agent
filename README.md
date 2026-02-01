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
git clone <your-repo-url>
cd bayesian-if-agent

# Install dependencies (includes Jericho and spacy model)
uv sync

# Download a game file
mkdir -p games
curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5
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
uv run python setup_check.py   # verify setup + quick 3-action demo
uv run python core.py           # unit tests for core components
uv run python runner.py         # full demo: walkthrough + 5-episode learning run
```

`runner.py` plays through 9:05 following the walkthrough (to verify setup), then trains the agent over 5 episodes with learning enabled, printing statistics about what it learned.

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

- **`core.py`** — All core components (GameState, StateParser, BeliefState, DynamicsModel, ActionSelector, BayesianIFAgent). Zero external dependencies — stdlib only.
- **`runner.py`** — Jericho integration, game-specific parser (`EnhancedStateParser`), experiment runner.

### Key Design Decisions

- **Frozen dataclasses** for `GameState` — hashable, usable as dict keys in transition counts, immutable for particle representation.
- **Count-based Bayesian inference** — `P(outcome|state,action) = count/total` with pseudocount prior (default 0.1). Simple, interpretable, online.
- **Thompson sampling** — sample state from belief, sample outcome from dynamics, pick best action. Naturally balances exploration/exploitation.
- **Particle filtering** — `Dict[GameState, float]` weights. Suited to discrete IF state spaces.

## Extending the Agent

### Custom State Parser

The default parser uses simple heuristics. Create a game-specific parser by subclassing `StateParser`:

```python
from core import StateParser, GameState

class ZorkParser(StateParser):
    def _extract_location(self, obs: str, prev) -> str:
        if 'west of house' in obs.lower():
            return 'west_of_house'
        return super()._extract_location(obs, prev)
```

See `EnhancedStateParser` in `runner.py` for a full example.

### LLM-Enhanced Parser

The parser is the natural integration point for an LLM:

```python
class LLMParser(StateParser):
    def __init__(self, llm_client):
        super().__init__()
        self.llm = llm_client

    def parse(self, observation: str, previous_state) -> GameState:
        prompt = f"Given this game observation:\n{observation}\n\n"
        prompt += "Extract: location, inventory, world state flags. Return as JSON."
        response = self.llm.complete(prompt)
        # Parse response into GameState
        ...
```

### Informed Priors

The dynamics model uses uniform priors by default. You can set `prior_pseudocount` per action type in `DynamicsModel` to encode domain knowledge (e.g., "take" actions succeed more often than "attack" actions).

## Troubleshooting

### "No module named 'jericho'"
Run `uv sync` to install dependencies.

### "FileNotFoundError: games/905.z5"
Download the game file — see Installation.

### Jericho build errors
Jericho requires gcc and make. On Ubuntu/Debian:
```bash
sudo apt-get install build-essential
```

## Related Work

- [Jericho](https://github.com/microsoft/jericho) — The IF game interface
- [TextWorld](https://github.com/microsoft/TextWorld) — Procedural IF generation
- [KG-A2C](https://arxiv.org/abs/2002.07626) — Knowledge graph RL for IF
- [TALES](https://microsoft.github.io/tale-suite/) — Text adventure benchmark suite

## Future Directions

1. **LLM Integration** — Use LLMs for state parsing and prior elicitation
2. **Structure Learning** — Learn which state variables matter (not just parameters)
3. **Meta-Learning** — Transfer learned dynamics across similar games
4. **Planning** — Look-ahead search under learned dynamics
5. **Natural Language Beliefs** — Represent beliefs as propositions, not just state vectors

## License

MIT
