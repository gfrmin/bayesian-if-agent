# bayesian-if-agent

Bayesian agent that learns to play text adventure games. Maintains probabilistic beliefs about game dynamics and uses expected utility maximization to select actions.

**Linux-only** (Jericho requirement).

## Setup

```bash
uv sync
mkdir -p games
curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5
```

## Running

```bash
uv run python setup_check.py   # verify deps + quick 3-action demo
uv run python core.py           # unit tests for core components
uv run python runner.py         # full demo: walkthrough + 5-episode learning run
```

Note: `runner.py` has a hardcoded game path (`/home/claude/games/905.z5` at line 370) that may need updating.

## Architecture

Five-component Bayesian pipeline, all in two files:

```
Observation → StateParser → BeliefState → DynamicsModel → ActionSelector → Action
                                    ↑                            |
                                    └── BayesianIFAgent ─────────┘
```

- **StateParser** (`core.py`): Converts raw text → `GameState(location, inventory, flags)`. Main LLM integration point — override `_extract_location`, `_extract_inventory`, `_extract_flags`.
- **BeliefState** (`core.py`): Particle filter over `GameState`. Represents P(current_state | observation_history).
- **DynamicsModel** (`core.py`): Count-based Bayesian inference. Learns P(next_state, reward | state, action) from observed transitions with pseudocount smoothing.
- **ActionSelector** (`core.py`): Thompson sampling + exploration bonus (expected utility + uncertainty). Balances exploration/exploitation.
- **BayesianIFAgent** (`core.py`): Orchestrates the pipeline. Stateless coordinator.

**JerichoRunner** (`runner.py`): Connects agent to Jericho/Frotz game environments. **EnhancedStateParser** (`runner.py`): Game-specific parser for 9:05.

## Key Design Decisions

- **Frozen dataclasses** for `GameState` — hashable, usable as dict keys in transition counts, immutable for particle representation.
- **Count-based Bayesian inference** — `P(outcome|state,action) = count/total` with pseudocount prior (default 0.1). Simple, interpretable, online.
- **Thompson sampling** — sample state from belief, sample outcome from dynamics, pick best action under sample. Naturally balances exploration/exploitation.
- **Particle filtering** — `Dict[GameState, float]` weights. Suited to discrete IF state spaces.
- **`core.py` has zero external deps** — only stdlib (`dataclasses`, `collections`, `random`, `math`, `json`). Jericho is only imported in `runner.py`.

## Extension Points

- **StateParser subclass**: Replace heuristic parsing with LLM calls for semantic state extraction.
- **Informed priors**: Set `prior_pseudocount` per action type in `DynamicsModel` (e.g., "take" actions succeed more often).
- **Structured state variables**: Currently flat (location, inventory, flags) — could learn which variables matter via meta-learning.

## Dependencies

`jericho>=3.0.0` (brings spacy transitively). No ML/RL frameworks.
