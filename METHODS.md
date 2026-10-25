# Method coverage and usage

This implements selected mechanisms from three distinct papers, not a complete
reproduction or a single algorithm equivalent to all three.

| Source | Implemented | Missing |
| --- | --- | --- |
| [Society of HiveMind](https://arxiv.org/pdf/2503.05473), Sec. 3.1.1 | Bernoulli edge distributions, REINFORCE, learned sigmoid baseline | Task-conditioned DistilBERT/GAT, continuous-distribution GA, hybrid gradient/evolution optimization |
| [Heterogeneous Swarms v2](https://arxiv.org/html/2502.04510v2), Algorithms 1, 2, 4 | Continuous adjacency PSO, G-Decode, frequency-weighted JFK attribution | Checkpoint/adapter weight updates and alternating role/weight optimization |
| [Xia et al., HiveMind: Contribution-Guided Online Prompt Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/40222), AAAI-26, Algorithms 1-4 | Exact Shapley for a zero-valued nonviable DAG game, structural coalition pruning, deterministic intermediate-output reuse, bottleneck targeting, reflection, node-specific prompt augmentation | Trading data, portfolio simulation, Sharpe utility, automatic collection/scheduling of trading windows, reproduced financial results |

## Running locally

The core uses only the Python standard library. From the project directory:

```bash
python -m pip install -e ".[dev]"
python examples/run_methods.py
pytest
```

The examples are offline fixtures. Their scores do not measure real model quality.
For real use construct `Task` objects and model specifications, then use an actual
backend. Supply actual provider prices yourself. No paid API calls are made by tests.

## Execution, caching and parallelism

`run_genome` prunes nodes that cannot reach the sink, then executes the rest one
DAG layer at a time (a node's layer is one more than its deepest included parent).
With `max_workers > 1` the nodes of a layer run in threads; results, nominal cost and
call counts match the sequential run exactly, and `outputs` is always keyed in node
order. `Evaluator(max_workers=N)` forwards this to every scoring and attribution call.
Backends must then be thread-safe; `CachedBackend`, `MockBackend` and `APIBackend` are.

`CachedBackend` keys on model name, endpoint, sample id and prompt. Prices are not part
of the key: a hit under changed prices is re-costed from stored token counts, so
correcting a price never re-spends. Generation settings (temperature, max_tokens) are
not part of the key either; use a different `cache_path` when you change them. With
`cache_path` set, every miss is appended as one JSON line and reloaded next run.

A consequence of caching worth knowing: two nodes with the same role, model, sample id
and inputs produce the same prompt and therefore the same cached text. If you want
independent samples from identical sibling agents, give them different roles or run
without the cache.

`APIBackend` retries transient failures (rate limits, timeouts, connection errors,
5xx, empty replies) with exponential backoff and lets everything else propagate at
once. Inside `run_genome` a failed call surfaces as `AgentError` carrying the node
index and model name; `BudgetExceeded` is passed through unchanged.

## Separate graph optimizers

```python
template = Genome([0, 1, 2], [0, 0, 1])
front = evolve(GAConfig(n_nodes=3), evaluator.objectives)
roles = search_roles(template, evaluator.utility)
policy = reinforce_graph(template, evaluator.utility)
credits = jfk_scores(roles.genome, len(evaluator.models), evaluator.utility)
```

NSGA-II minimizes accuracy error and mean nominal inference USD/task and returns a
Pareto front. PSO and REINFORCE maximize scalar utility; default `Evaluator.utility`
is accuracy. An explicit `cost_weight` gives accuracy minus a dollar-cost penalty.

REINFORCE learns independent Bernoulli edge logits for a fixed topological order.
The old baseline is used for the whole batch before baseline fitting, avoiding
action-dependent batch-baseline bias. The returned distribution can be sampled;
`distribution.threshold()` is a separate deterministic view. The best observed
graph need not be the thresholded final distribution. Template edges are not an
initial policy: role and model identities are fixed, edge logits start near zero.

PSO uses normalized inertia, cognitive, social, and global-worst repulsion terms.
G-Decode selects a sink by inverse out-degree, adds nodes using out-degree top-p
sampling, and samples edges to already added nodes using softmax probabilities.
Relabeling preserves role/model identities and the harness's forward-index DAG
convention. Nonpositive row sums are floored at 1e-12 for degree sampling; this is
an explicit numerical policy beyond the paper's equations. There is no clipping
of particle coordinates. All scalar utilities are maximized.

Because decoding is stochastic, `search_roles` retains the actual best observed
graph, its measured utility, and its source matrix. A fresh decode may differ.
`decode_samples` averages multiple decodes for particle evaluation; result utility
is the best individual observation, not a mean particle score.

JFK scores weight utility by the expert's occurrence count in random assignments.
Unsampled experts receive `None`. This is contribution measurement, not checkpoint
training. Full Heterogeneous Swarms needs compatible open-weight experts, actual
parameter-space updates, storage, and alternation with role search. The hosted
API backend does not expose model weights.

## Coalition attribution and information flow

`Evaluator.attribute(graph, deterministic=True)` uses mean task score as coalition
utility. `dag_shapley(graph, value_of_coalition)` accepts a custom utility such as
a separately implemented trading simulator's Sharpe ratio.

Coalitions lacking the original sink or an original source-to-sink path have zero
utility **by definition of this game**. A missing original source is not replaced
by an intermediate node whose parents are absent. This assumption must suit the
task: even a standalone sink capable of answering a question is assigned zero
if it violates the source rule. Attribution utility therefore differs from
ordinary unconstrained `Evaluator.score`.

Coalition execution removes absent nodes and incident edges and permits partial
inputs. `run_genome` prunes nodes that cannot reach the sink and returns node
outputs for experience collection. By default every node sees the task. Pass
`external_inputs={node: raw_data}` to expose raw data only to named nodes; other
nodes receive their role instructions and parent outputs. Through
`Evaluator.attribute`, supply a task-to-mapping callback as `external_inputs`.

Output reuse assumes deterministic executions under a fixed backend/checkpoint
and prompt configuration. `deterministic=True` is a caller assertion, not a
detector. A frozen stochastic cache does not prove exact attribution of expected
stochastic performance. Node caches are fresh per attribution call, avoiding reuse
after prompt or checkpoint updates. Enumeration and the Shapley sum remain
exponential. Default `max_nodes=16` guards against accidental large workloads.

Tests reproduce 49 viable coalitions and 73 agent calls for the seven-node
`[3,3,1]` DAG and compare attribution against exhaustive Shapley on the same game.
This validates the algorithmic mechanism, not the paper's trading outcomes.

## Contribution-guided prompt cycles

```python
update = optimize_prompts(
    graph, evaluator.roles, history,
    lambda g: evaluator.attribute(g, deterministic=True),
    reflector_backend, reflector_model, threshold=0.1,
)
graph = update.genome
evaluator.roles = update.roles
```

Supply `Experience(node, state, action, reward)` from the observed window. Negative
rewards are failures; nonnegative rewards are successes. Accuracy users must
choose an explicit mapping (e.g. correct = +1 and incorrect = -1). Node outputs
come from `run_genome(...).outputs`; state and outcome rewards are caller supplied.
A global task reward is not claimed to be an individual causal reward.

Each cycle measures contributions, selects the lowest-scoring node, and calls the
reflection backend only if the score is below threshold and history exists. The
lessons augment that node's prompt; shared role indices are split so other nodes
keep their instructions. Install both the returned graph and role table before
the next window. The default threshold is zero; positive-credit agents require
an appropriately chosen positive threshold to trigger refinement. This is a
callable cycle, not a scheduler, and prompt augmentation does not guarantee gains.
Use separate validation data to assess changes before consulting the final test.

## Held-out evaluation

`split_tasks` groups identical prompts, rejects conflicting references, and creates
train/validation/test sets. For already sorted temporal data, use
`chronological=True`; unique prompts are required to preserve ordering.

Search on train, select on validation, and call `evaluate_frozen` on an untouched
test evaluator. Supply every consulted train/validation task as `search_tasks`.
Exact-prompt overlap is rejected. Candidate order is preserved; test scores are
not used to select a winner. This cannot detect paraphrases, external dataset
leakage, or tasks omitted from search history. Repeated samples use distinct cache
IDs; hosted APIs need not expose deterministic random seeds.

Report repeated runs, held-out scores, search spend, and inference spend separately.
Baseline APIs report costs but do not enforce equal spending. `CachedBackend`
checks recorded spend before starting a call, so one response can overshoot its
cap. It is not a strict pre-authorized monetary bound.
