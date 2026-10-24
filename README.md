# agentswarm

I read a few papers about LLM swarms and wanted to try building one myself.
This is my attempt at reproducing parts of their methods and putting them in a
small Python project I can mess around with.

**My swarm worked.** I ran a four-agent workflow through OpenRouter on a group
theory question. The agents passed their responses along and the final answer
was correct. One routed model returned an irrelevant classification, so there
are definitely rough edges. That's the demo so far, not a benchmark.

## What's in here

- Agent graphs that pass responses between models.
- Genetic search, PSO graph search, and REINFORCE for experimenting with connections.
- DAG-Shapley and JFK contribution scoring.
- A prompt-update cycle that reflects on an agent's past failures and successes.
- A LiteLLM backend, so you can use different model APIs or a compatible local endpoint.
- Offline examples, baseline helpers, and train/validation/test splitting.

The package is still called `evoharness` in Python.

## Try it

```bash
git clone https://github.com/Co973/agentswarm.git
cd agentswarm
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -e ".[api]"
```

Copy `.env.example` to `.env`. Paste your API key into **your local `.env`**.
The example defaults to OpenRouter's free router, so you don't have to choose
a particular model. You'll need an OpenRouter key; availability and rate limits
still apply. Keep your `.env` private -- it's ignored by Git.

```dotenv
LLM_API_KEY=your-openrouter-key
LLM_MODEL=openai/openrouter/free
LLM_API_BASE=https://openrouter.ai/api/v1
```

Then give the swarm something to do:

```bash
python examples/run_swarm.py --prompt "Your question here" --temperature omit --max-tokens 8192
```

That runs the fixed four-agent demo. To try graph optimization and prompt
updates offline, run `python examples/run_methods.py`. The original genetic
search demo is `python examples/run_mock.py`. Those two use fixtures/mock
responses, so their scores aren't real model results.

You can change the model and endpoint in `.env` for another provider.
[LiteLLM's docs](https://docs.litellm.ai/docs/providers) explain the provider/model
names. Cost reporting is off unless you supply `--price-in` and `--price-out`
in USD per million tokens; an auto router doesn't have one fixed model price.

## Papers I took inspiration from

Credit for the research and methodology goes to the authors of these papers:

- [The Society of HiveMind: Multi-Agent Optimization of Foundation Model Swarms to Unlock the Potential of Collective Intelligence](https://arxiv.org/pdf/2503.05473)
  -- Noah Mamie and Susie Xi Rao. Inspired the graph-distribution learning and evolutionary side.
- [Heterogeneous Swarms: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](https://arxiv.org/html/2502.04510v2)
  -- Shangbin Feng et al. Inspired PSO graph search, G-Decode, and JFK contribution scoring.
- [HiveMind: Contribution-Guided Online Prompt Optimization of LLM Multi-Agent Systems](https://ojs.aaai.org/index.php/AAAI/article/view/40222)
  -- Yihan Xia et al. Inspired DAG-Shapley and contribution-guided prompt updates.

This is a partial reproduction attempt, not the authors' implementation.
I haven't reproduced their published results. Model-weight optimization,
task-conditioned GAT routing, and the trading experiments aren't implemented.
[METHODS.md](METHODS.md) has the details if you want to dig into what matches
and what doesn't.

## If you want to work on it

I haven't done proper real-model benchmarking or tested this broadly across
tasks. There are local tests in `tests/`; you're welcome to run them, break
things, add your own evaluations, and work out whether the swarm helps on
your tasks. Issues and PRs are welcome.

```bash
python -m unittest discover -s tests -v
```

MIT licensed. Experiment with it.
