# bayesian-if-agent

Bayesian agent that learns to play text adventure games via expected utility maximization.

## Commands

```bash
uv sync                         # install deps
uv run python setup_check.py    # verify deps + quick 3-action demo
uv run python core.py            # unit tests for core components
uv run python sensor_model.py    # unit tests for LLM sensor model (deprecated)
uv run python runner.py          # full demo: walkthrough + 5-episode learning run
uv run python runner.py --episodes 3 --verbose  # verbose learning run
```

Game files go in `games/`. Download 9:05: `curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5`

## Architecture

Flat file layout. `core.py` is stdlib-only (no external deps). `runner.py` imports Jericho. LLM code in separate files.

**Pipeline**: Jericho ground truth -> AgentBeliefs -> InformedActionSelector (+ LLM oracle) -> DynamicsModel -> Action

`core.py` components (stdlib-only):
- **GameState** — frozen dataclass (location: int, inventory: frozenset, world_hash: str). Hashable, used as dict keys.
- **Outcome** — frozen dataclass (next_state, reward).
- **Transition** — frozen dataclass for history tracking.
- **AgentBeliefs** — mutable dataclass: location, inventory, goals, blockers, failed actions, tracked state, history. Has `to_prompt_context()` and `to_state_key() -> GameState`.
- **SituationUnderstanding** — dataclass: LLM's structured analysis (goals, blockers, recommendations). Has `from_json()` classmethod.
- **OracleReliability** — dataclass: tracks recommendation/goal/blocker accuracy from experience.
- **DynamicsModel** — count-based P(next_state, reward | state, action) with pseudocount prior.
- **BeliefUpdater** — updates AgentBeliefs from oracle observations (understanding, failure, progress).
- **InformedActionSelector** — hierarchical: oracle recommendation -> blocker addressing -> dynamics -> alternatives -> random. Oracle injected (no imports).
- **BayesianIFAgent** — orchestrator. Maintains beliefs, reliability, dynamics, selector. Oracle injected at construction.

LLM oracle files:
- **`oracle.py`** — LLMOracle: analyses situation, failure, progress; suggests state variables; extracts state. Requires `requests` via OllamaClient.
- **`ollama_client.py`** — OllamaClient: HTTP interface to local Ollama with MD5 cache. Requires `requests`.
- **`sensor_model.py`** — (Deprecated) LLMSensorModel: Beta-distributed TPR/FPR. Kept for backward compatibility.

`runner.py` components:
- **`extract_state(env)`** — extracts GameState from Jericho ground truth.
- **`extract_beliefs_data(env)`** — extracts location name, ID, inventory list, world hash for belief updates.
- **`check_ollama()`** — checks if Ollama is running.
- **JerichoRunner** — bridges Jericho to agent. Detects oracle availability at startup. argparse CLI.

`metareason.py` — V(think) vs V(act) deliberation budget.
`contradiction.py` — detects (state, action) pairs with multiple outcomes (diagnostic).

## LLM Oracle Setup

The LLM oracle is **optional**. Without it, the agent uses dynamics-only + random exploration.

```bash
# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve  # runs on localhost:11434
```

When `runner.py` starts, it auto-detects Ollama and enables/disables the oracle accordingly.

## Constraints

- **Linux-only** (Jericho requirement).
- `GameState` must stay frozen/hashable — used as dict keys in transition counts.
- `core.py` and `sensor_model.py` must remain stdlib-only (no external imports).
- Oracle objects are injected into core classes, never imported directly.
- Prefer functional style.
