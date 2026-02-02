# bayesian-if-agent

Bayesian agent that learns to play text adventure games via expected utility maximization.

## Commands

```bash
uv sync                         # install deps
uv run python setup_check.py    # verify deps + quick 3-action demo
uv run python core.py            # unit tests for core components
uv run python sensor_model.py    # unit tests for LLM sensor model
uv run python runner.py          # full demo: walkthrough + 5-episode learning run
```

Game files go in `games/`. Download 9:05: `curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5`

## Architecture

Flat file layout. `core.py` is stdlib-only (no external deps). `runner.py` imports Jericho. LLM code in separate files.

**Pipeline**: Jericho ground truth → GameState → BayesianActionSelector (+ LLM sensor) → DynamicsModel → Action

`core.py` components (stdlib-only):
- **GameState** — frozen dataclass (location: int, inventory: frozenset, world_hash: str). Hashable, used as dict keys.
- **Outcome** — frozen dataclass (next_state, reward).
- **Transition** — frozen dataclass for history tracking.
- **DynamicsModel** — count-based P(next_state, reward | state, action) with pseudocount prior.
- **BayesianActionSelector** — Thompson sampling + LLM sensor prior + exploration bonus. Sensor/sensor_model injected (no imports).
- **BayesianIFAgent** — orchestrator. State set externally from Jericho.

LLM sensor files:
- **`sensor_model.py`** — LLMSensorModel: Beta-distributed TPR/FPR, learns LLM reliability from experience. Stdlib-only.
- **`ollama_client.py`** — OllamaClient: HTTP interface to local Ollama with MD5 cache. Requires `requests`.
- **`action_sensor.py`** — LLMActionSensor (queries Ollama) + UniformActionSensor (fallback). Requires `requests`.

`runner.py` components:
- **`extract_state(env)`** — extracts GameState from Jericho ground truth (location ID, inventory, world hash).
- **JerichoRunner** — connects agent to Jericho/Frotz environments. Detects Ollama availability at startup.

`metareason.py` — V(think) vs V(act) deliberation budget.
`contradiction.py` — detects (state, action) pairs with multiple outcomes (diagnostic).

## LLM Sensor Setup

The LLM sensor is **optional**. Without it, the agent uses uniform priors (exploration-only).

```bash
# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
ollama serve  # runs on localhost:11434
```

When `runner.py` starts, it auto-detects Ollama and enables/disables the LLM sensor accordingly.

## Constraints

- **Linux-only** (Jericho requirement).
- `GameState` must stay frozen/hashable — used as dict keys in transition counts.
- `core.py` and `sensor_model.py` must remain stdlib-only (no external imports).
- LLM sensor objects are injected into core classes, never imported directly.
- Prefer functional style.
