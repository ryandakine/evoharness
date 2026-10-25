# evoharness

A small, dependency-free Python harness for building **graphs of LLM agents** and then
optimizing them: which agents to use, how to wire them together, and what each one
should be told. It reproduces mechanisms from three LLM-swarm papers (see
[Credits](#credits)) and adds the plumbing needed to actually run them against real
models: cost tracking, response caching, budgets, retries, and a command line.

```
$ evoharness --prompt "A train leaves at 3:40pm and arrives at 6:05pm. How long is the trip?"

--- Agent 0 [openai/gpt-4o-mini]: Solve the task carefully and explain your solution. ---
...
--- Agent 3 [openai/gpt-4o-mini]: Produce the final response. Finish with 'Answer: <final answer>'. ---
...
FINAL: 2 hours 25 minutes
Model calls: 4 | elapsed 9.8s
```

What is in the box:

- **Agent graphs** (`Genome`): a DAG where each node is a role prompt plus a model, and
  edges pass one agent's output into the next agent's context. Save and load them as JSON.
- **Runner** (`run_genome`): executes a graph on a task, in parallel where the graph
  allows, and reports the answer, every intermediate output, call count and nominal cost.
- **Optimizers**: NSGA-II genetic search over accuracy *and* cost (`evolve`), particle
  swarm search over continuous adjacency matrices with G-Decode (`search_roles`), and
  REINFORCE over Bernoulli edge distributions (`reinforce_graph`).
- **Attribution**: exact DAG-Shapley contribution scores (`dag_shapley`) and
  frequency-weighted JFK model scores (`jfk_scores`), so you can see which agent is
  pulling its weight.
- **Prompt refinement** (`optimize_prompts`): a reflection cycle that finds the weakest
  agent and appends lessons from its own failures to its instructions.
- **Backends**: anything LiteLLM supports (OpenAI, Anthropic, OpenRouter, Ollama, vLLM,
  ...), a persistent cache with a spend cap, and a deterministic mock for offline work.
- **Evaluation**: leakage-guarded train/validation/test splits and frozen held-out scoring.

## Install

```bash
git clone https://github.com/surren83-ctrl/evoharness.git
cd evoharness
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[api]"
```

The core library has no dependencies. The `[api]` extra adds LiteLLM and python-dotenv
for talking to real models.

## Run a swarm

Copy `.env.example` to `.env` and put your key in it (the file is git-ignored):

```
LLM_API_KEY=your-key
LLM_MODEL=openai/gpt-4o-mini
LLM_API_BASE=
```

`LLM_MODEL` is a LiteLLM name: `provider/model`, or `openai/<model-id>` plus an
`LLM_API_BASE` for any OpenAI-compatible server (OpenRouter, Ollama, vLLM, LM Studio).
[LiteLLM's provider list](https://docs.litellm.ai/docs/providers) has the names.

```bash
evoharness --prompt "Your question here"                    # default 'debate' swarm, 4 agents
evoharness --preset committee --prompt "..."                # three solvers + a judge
evoharness --preset pipeline --prompt "..." --json          # machine-readable report
evoharness --spec examples/swarm_spec.json --prompt "..."   # your own graph
evoharness --list-presets
```

Useful flags: `--parallel N` (independent agents run concurrently, default 2),
`--retries N` (transient errors and empty replies are retried with backoff, default 2),
`--max-tokens`, `--temperature` (or `omit` for models that reject them), `--timeout`,
and `--price-in` / `--price-out` in USD per million tokens to get a cost estimate.

`examples/run_swarm.py` and `run_swarm.ps1` are thin wrappers around the same command.

### Writing your own swarm

A spec is a JSON file with the roles, one node per agent, and edges from lower to
higher node index. The last node produces the answer; it should end with
`Answer: <final answer>` so the harness can extract it.

```json
{
  "roles": ["Solve the task and show your work.",
            "Check the supplied solution for mistakes.",
            "Write the final response. Finish with 'Answer: <final answer>'."],
  "nodes": [{"role": 0}, {"role": 1}, {"role": 2}],
  "edges": [[0, 1], [1, 2], [0, 2]],
  "models": ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4-5"]
}
```

`models` is optional; without it every agent uses `--model`. With it, a node can say
`"model": 1` to pick the second one. Entries can also be objects with `name`,
`price_in`, `price_out`, `api_base` and `api_key_env` for mixed-provider swarms.

### A note on free routers

`openai/openrouter/free` style auto-routers send each call to whatever free model is
available at that moment. Different agents in one swarm can land on different models
with different habits, and empty replies under load are common. The harness retries
empties, but for anything you want to reason about, pin a specific model.

## Optimize a swarm

The offline demos use fixture backends, so their numbers say nothing about real model
quality; they show the mechanics end to end:

```bash
python examples/run_mock.py      # NSGA-II: Pareto front of accuracy vs. cost
python examples/run_methods.py   # PSO, REINFORCE, JFK, DAG-Shapley, prompt reflection, held-out eval
```

With a real backend the pieces fit together like this:

```python
from evoharness import APIBackend, CachedBackend, Evaluator, GAConfig, ModelSpec, Task, evolve

tasks = [Task("What is 17 * 23?", "391"), ...]                 # prompt + reference answer
models = [ModelSpec("openai/gpt-4o-mini", 0.15, 0.60),          # USD per 1M tokens in/out
          ModelSpec("openai/gpt-4o", 2.50, 10.00)]
backend = CachedBackend(APIBackend(), budget_usd=5.0, cache_path="cache.jsonl")
ev = Evaluator(tasks, backend, models, max_workers=4)

front = evolve(GAConfig(n_nodes=4, n_models=2, pop_size=12, generations=6), ev.objectives)
for genome, (error, cost) in front.front:
    print(f"acc={1 - error:.2f}  ${cost:.5f}/task  {genome.to_dict()}")

credit = ev.attribute(front.front[-1][0], deterministic=True)   # who contributed?
```

The disk cache means a crashed or repeated search re-spends nothing, and the budget cap
stops a runaway search. `METHODS.md` explains each method, what matches the papers and
what does not.

## Development

```bash
python -m pip install -e ".[api,dev]"
pytest            # 60+ tests, all offline; no API calls
ruff check .
```

Tests and lint run in CI on Python 3.10 through 3.13. Issues and PRs are welcome.

## Credits

This is a fork of [surren83-ctrl/evoharness](https://github.com/surren83-ctrl/evoharness)
(originally published as `agentswarm`), a learning project reproducing parts of:

- [The Society of HiveMind: Multi-Agent Optimization of Foundation Model Swarms](https://arxiv.org/pdf/2503.05473) -- Noah Mamie and Susie Xi Rao. Graph-distribution learning and the evolutionary side.
- [Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](https://arxiv.org/html/2502.04510v2) -- Shangbin Feng et al. PSO graph search, G-Decode, and JFK contribution scoring.
- [HiveMind: Contribution-Guided Online Prompt Optimization of LLM Multi-Agent Systems](https://ojs.aaai.org/index.php/AAAI/article/view/40222) -- Yihan Xia et al. DAG-Shapley and contribution-guided prompt updates.

It is a partial reproduction, not the authors' implementation, and their published
results have not been reproduced here. Model-weight optimization, task-conditioned GAT
routing and the trading experiments are not implemented. See `CHANGELOG.md` for what
this fork changed.

MIT licensed.
