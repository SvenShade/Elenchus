# Elenchus

**Elenchus, Lantern of Latent Labyrinths v0.1** is an experimental
LLM-plus-MCTS system for dialectic self-exploration.

Underneath every conversation lies an unspoken labyrinth. Elenchus enters these
labyrinths as a lantern, testing Doors, gathering Keys, and following Passages
into Chambers of simulated dialogue. It observes where candor sharpens or
retreats, then returns with words carefully chosen.

The core idea is to avoid planning directly in token space. Instead, Elenchus
plans over a lifted strategic action space defined in YAML. An LLM supplies
priors, imagined conversation continuations, cognitive-state analysis, and
utility judgements. MCTS supplies disciplined search, exploration, backup, and
traceability.

## What Is Here

- A Python 3.11 package and CLI: `llm-mcts`.
- A React, Three.js, and FastAPI visual UI.
- A local OpenAI-compatible LLM transport, designed for vLLM.
- YAML-defined two-player conversation games.
- The Elenchus game: `examples/elenchus_v0_1.yaml`.
- A generic starter game template: `examples/template.yaml`.
- Progressive widening over generated action candidates.
- Belief-graph extraction, latent user hypotheses, and abstract search stats.
- Reflexion mode: a first tree produces private Cartography, then a second tree
  replans with that evidence.
- JSONL LLM call traces and `tree.json` search dumps under `runs/`.

## Labyrinth Vocabulary

The UI keeps MCTS technical state visible through a consistent metaphor:

- **Chamber**: an MCTS node, a reached conversational state.
- **Door**: a lifted action candidate that may be widened into the tree.
- **Keyring**: the progressive widening limit.
- **Key**: one available child slot admitted by widening.
- **Passage**: an edge or transition between Chambers.
- **Route**: the selected root-to-Chamber chain.
- **Floorplan**: the extracted belief graph.
- **Lockwork**: factorized action-space statistics.
- **Omen**: an LLM evaluation.
- **Lantern**: human-readable value estimate.
- **Cartography**: private rollout reflection before speaking.

## Quick Start: Mock UI

This mode needs no live model and is the fastest way to inspect the interface.

```bash
uv sync --extra dev
npm --prefix frontend install
uv run --extra web llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --mock-llm
```

Open the printed Vite URL, usually:

```text
http://127.0.0.1:5173
```

## Quick Start: Local vLLM

The web command starts or reuses an OpenAI-compatible local LLM server, starts
the FastAPI backend, starts the Vite frontend, and prints the browser URL.

For the current 2x RTX 2080 Ti workstation:

```bash
uv sync --extra web --extra local-llm
npm --prefix frontend install
uv run --extra web --extra local-llm llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --preset 2080ti-qwen-awq-tp2
```

Useful presets:

```bash
# Very small smoke-test model.
uv run --extra web --extra local-llm llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --preset 2080ti-qwen-smoke

# Single-card Qwen2.5 7B AWQ.
uv run --extra web --extra local-llm llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --preset 2080ti-qwen-awq

# Two-card tensor-parallel Qwen2.5 7B AWQ.
uv run --extra web --extra local-llm llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --preset 2080ti-qwen-awq-tp2
```

You can also point at an already running OpenAI-compatible server:

```bash
uv run --extra web llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --base-url http://127.0.0.1:8000/v1 \
  --model your-model \
  --no-vllm
```

## CLI Usage

```bash
uv run llm-mcts validate --game examples/elenchus_v0_1.yaml
uv run llm-mcts plan --game examples/elenchus_v0_1.yaml --mock-llm
uv run llm-mcts play --game examples/elenchus_v0_1.yaml --mock-llm --p2-mode manual
```

The CLI and web UI both use the same backend planner.

## YAML Games

Games live in `examples/` and are loaded by the general MCTS plumbing.
To define your own game, copy:

```text
examples/template.yaml
```

A game YAML defines:

- `metadata`: name, description, max real turns.
- `llm`: base URL, model, sampling, completion transport, timeouts.
- `players`: P1 and P2 public/private profiles and beliefs.
- `actions`: lifted strategic action cards.
- `prompts`: priors, state analysis, candidate generation, turn continuation,
  rollout judgement, Cartography, finalization, and actual P2 replies.
- `mcts`: simulation budget, Labyrinth/tree depth, Lantern/rollout range, PUCT,
  progressive widening.
- `initial_state`: task and optional starting transcript.

Elenchus uses natural transcript continuations for P1/P2 utterance generation
and JSON for priors, judges, state analysis, candidates, and reflection.

## Architecture

At each P1 move:

1. The environment formats the public transcript and P1-visible private context.
2. The LLM extracts a typed cognitive state and belief graph.
3. The LLM proposes lifted Door candidates from action cards and state targets.
4. Progressive widening admits a bounded number of Doors into the MCTS tree.
5. MCTS selects using PUCT over expanded children.
6. P1 continuations and imagined P2 replies are generated as transcript
   completions.
7. The LLM judges reached rollout states with a scalar utility and rubric.
8. Utilities are backed up into local tree stats and abstract factor stats.
9. Optional Cartography reflects on explored and deferred evidence.
10. The final P1 utterance is produced and appended to the real chat.

Partial observability is explicit: actual P2 prompts see public context only.
P1 planning, state analysis, candidates, judges, reflection, and finalization may
see P1-private context.

## Development

Python tests:

```bash
uv run pytest
```

Frontend unit tests and build:

```bash
npm --prefix frontend test -- --run
npm --prefix frontend run build
```

End-to-end UI smoke test:

```bash
uv run --extra web llm-mcts web \
  --game examples/elenchus_v0_1.yaml \
  --mock-llm \
  --no-open

npm --prefix frontend run test:e2e
```

## Traces

Planner runs write research artifacts under `runs/<timestamp>/`:

- `calls.jsonl`: every LLM request and response, including parse errors.
- `tree.json`: MCTS nodes, candidates, visits, values, utterances, rollouts,
  evaluations, root stats, chosen action, and reflection.

`runs/` is intentionally ignored by git.
