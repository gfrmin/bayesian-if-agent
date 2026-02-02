# bayesian-if-agent

Bayesian agent that learns to play text adventure games via expected utility maximization.

## Commands

```bash
uv sync                         # install deps
uv run python setup_check.py    # verify deps + quick 3-action demo
uv run python core.py            # unit tests for core components
uv run pytest tests/             # full test suite
uv run python runner.py          # full 10-episode learning run
uv run python runner.py --episodes 3 --verbose  # verbose learning run
uv run python runner.py --question-cost 0.05     # adjust LLM query cost
uv run python runner.py --action-cost 0.15       # increase cost of taking actions
uv run python runner.py --action-prior 0.10      # more conservative action prior
```

Game files go in `games/`. Download 9:05: `curl -L "https://www.ifarchive.org/if-archive/games/zcode/905.z5" -o games/905.z5`

## Architecture

Flat file layout. `core.py` is stdlib-only (no external deps). `runner.py` imports Jericho. LLM code in `ollama_client.py`.

**Pipeline**: Game observation -> LLM sensor bank (binary yes/no questions) -> BeliefState -> UnifiedDecisionMaker (EU maximization) -> Action

`core.py` components (stdlib-only):
- **BinarySensor** — Beta-distributed TPR/FPR. Learns reliability from ground truth. Computes posterior via Bayes' rule.
- **QuestionType** — Enum: ACTION_HELPS, IN_LOCATION, HAVE_ITEM, STATE_FLAG, GOAL_DONE, PREREQ_MET, ACTION_POSSIBLE.
- **LLMSensorBank** — Per-type BinarySensors + LLM queries. Duck-typed `llm_client` injected (any `.complete(prompt) -> str`). Per-turn query cache.
- **BeliefState** — Probability dicts for location, inventory, flags, goals, actions. Updated via Bayes' rule. `to_context_string()` for LLM context.
- **StateActionKey** — Frozen dataclass (state_hash, action). Hashable, used as dict keys.
- **ObservedOutcome** — Dataclass (next_state_hash, reward, observation_text).
- **DynamicsModel** — Deterministic: one observation = certainty. Maps StateActionKey -> ObservedOutcome.
- **UnifiedDecisionMaker** — VOI-based ask-or-act. Computes EU for game actions and questions in same framework. `action_cost` penalizes taking actions (limited turns), `action_prior` sets conservative default belief (most IF actions don't help).

`runner.py` components:
- **BayesianIFAgent** — Orchestrator. Unified ask-or-act loop, dynamics learning, sensor reliability learning. Optional LLM (dynamics-only fallback).
- **check_ollama()** — Checks if Ollama is running.
- **main()** — argparse CLI with game, episodes, max-steps, verbose, model, question-cost, action-cost, action-prior.

`ollama_client.py`:
- **OllamaConfig** — Dataclass: model, base_url, temperature, timeout.
- **OllamaClient** — HTTP interface to local Ollama. Single method: `.complete(prompt) -> str`.

## LLM Sensor Setup

The LLM sensor bank is **optional**. Without it, the agent uses dynamics-only + uniform priors.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:latest
ollama serve  # runs on localhost:11434
```

When `runner.py` starts, it auto-detects Ollama and enables/disables the sensor bank accordingly.

## Constraints

- **Linux-only** (Jericho requirement).
- `core.py` must remain stdlib-only (no external imports). LLM client injected via duck typing.
- `StateActionKey` is frozen/hashable — used as dict keys in dynamics.
- Prefer functional style.
