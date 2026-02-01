# bayesian-if-agent

Bayesian agent that learns to play text adventure games via expected utility maximization.

## Commands

```bash
uv sync                         # install deps
uv run python setup_check.py    # verify deps + quick 3-action demo
uv run python core.py            # unit tests for core components
uv run python runner.py          # full demo: walkthrough + 5-episode learning run
```

Game files go in `games/`. Download 9:05: `curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5`

**Gotcha**: `runner.py:370` has a hardcoded game path (`/home/claude/games/905.z5`) — update it before running.

## Architecture

Two files. `core.py` is stdlib-only (no external deps). `runner.py` imports Jericho.

**Pipeline**: Observation → StateParser → BeliefState → DynamicsModel → ActionSelector → Action

`core.py` components:
- **GameState** — frozen dataclass (location, inventory, flags). Hashable, used as dict keys.
- **StateParser** — text → GameState. Override `_extract_location`, `_extract_inventory`, `_extract_flags`.
- **BeliefState** — particle filter, `Dict[GameState, float]` weights.
- **DynamicsModel** — count-based P(next_state, reward | state, action) with pseudocount prior.
- **ActionSelector** — Thompson sampling + exploration bonus.
- **BayesianIFAgent** — stateless orchestrator.

`runner.py` components:
- **JerichoRunner** — connects agent to Jericho/Frotz environments.
- **EnhancedStateParser** — game-specific StateParser subclass for 9:05 (reference implementation for custom parsers).

## Constraints

- **Linux-only** (Jericho requirement).
- `GameState` must stay frozen/hashable — used as dict keys in transition counts and particle weights.
- `core.py` must remain stdlib-only (no external imports).
- Prefer functional style.

## Extension Pattern

Subclass `StateParser` and override the `_extract_*` methods. See `EnhancedStateParser` in `runner.py` as the reference example. The parser is the main LLM integration point.
