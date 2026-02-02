# bayesian-if-agent

Bayesian agent for Interactive Fiction — all decisions via expected utility maximisation. LLM is a queryable sensor with learnable reliability.

## Design Principles (NON-NEGOTIABLE)

1. **Everything is argmax EU.** Actions, questions, which question — all EU maximisation. No special cases. If you're writing `if stuck` or `with probability p` — STOP, derive from EU.
2. **No hacks.** No exploration bonuses, loop detection, novelty rewards, or "if stuck do X". Fix the model, not symptoms.
3. **LLM outputs are data.** Sensor with learnable TPR/FPR. Updates beliefs via Bayes' rule. Never follow blindly.
4. **Direct experience is certain.** Deterministic game + observed outcome = certainty. No need to query LLM about known outcomes.
5. **Justify every parameter.** No magic numbers. If you can't explain it, remove it.
6. **When stuck, derive the math.** Write EU of each option. Find what's broken. Fix it.

## Key Files

- `SPEC.md` — Full specification with math derivations and pseudocode
- `core.py` — Core components (stdlib-only): BinarySensor, LLMSensorBank, BeliefState, DynamicsModel, UnifiedDecisionMaker
- `runner.py` — BayesianIFAgent orchestrator + CLI (imports Jericho)
- `ollama_client.py` — HTTP interface to local Ollama
- `setup_check.py` — Dependency verification + quick 3-action demo
- `tests/` — test_core.py, test_agent.py, test_integration.py

## Architecture

Flat file layout. `core.py` is stdlib-only. LLM client injected via duck typing.

**Pipeline:** Game observation → LLM sensor bank (binary yes/no questions) → BeliefState → UnifiedDecisionMaker (EU maximisation) → Action

- **BinarySensor** — Beta-distributed TPR/FPR, learns from ground truth, computes posterior via Bayes' rule
- **LLMSensorBank** — Per-question-type sensors + LLM queries, per-turn cache
- **BeliefState** — Probability dicts for location, inventory, flags, goals, actions
- **DynamicsModel** — Deterministic: one observation = certainty
- **UnifiedDecisionMaker** — VOI-based ask-or-act in unified EU framework

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

## LLM Sensor Setup

Optional — without it, agent uses dynamics-only + uniform priors.

Any llama3.1 variant works (8b, 70b, etc) — the Bayesian framework learns any sensor's reliability from data.

```bash
ollama pull llama3.1:latest
ollama serve  # localhost:11434
```

Auto-detected by `runner.py` at startup.

## Debugging

1. Print VOI for potential questions — is it > cost?
2. Print beliefs — reasonable? Prior mean should be ~1/N
3. Check ground truth — only `reward > 0` counts as "helped"
4. Check LLM responses — is it being queried? What does it say?
5. Working agent: asks when VOI > cost, updates beliefs, learns sensor reliability, doesn't loop

**Never add a hack. Find the root cause.**

## Constraints

- Linux-only (Jericho requirement)
- `core.py` must remain stdlib-only — LLM client injected via duck typing
- `StateActionKey` is frozen/hashable — used as dict keys
- Prefer functional style
