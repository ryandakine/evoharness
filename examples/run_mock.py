"""Offline demo using the mock backend. Swap in APIBackend for real use.

Prices below are placeholders for the demo only. Look up real provider pricing.
"""
from evoharness import (
    CachedBackend,
    Evaluator,
    GAConfig,
    MockBackend,
    ModelSpec,
    Task,
    evolve,
    majority_vote,
    single_agent,
)

tasks = [Task(f"question-{i}", f"ans-{i}") for i in range(30)]
models = [ModelSpec("small", 0.25, 1.25), ModelSpec("large", 3.0, 15.0)]
mock = MockBackend({t.prompt: t.reference for t in tasks}, base=[0.45, 0.70], model_names=[m.name for m in models])
backend = CachedBackend(mock, budget_usd=5.0)
ev = Evaluator(tasks, backend, models)

print("Baselines (accuracy, $/task):")
for i, m in enumerate(models):
    print(f"  single {m.name:6s}", ev.score(single_agent(i)))
    print(f"  vote5  {m.name:6s}", majority_vote(ev, i, 5))

res = evolve(GAConfig(n_nodes=4, n_models=2, pop_size=16, generations=8), ev.objectives)
print("\nPareto front (accuracy, $/task):")
for g, (err, cost) in res.front:
    names = [models[g.models[i]].name for i in g.active_nodes()]
    print(f"  acc={1-err:.2f} cost=${cost:.6f} nodes={g.active_nodes()} models={names}")
print(f"\ncache hits={backend.hits} misses={backend.misses} real spend=${backend.spent:.4f}")
print("\nBest graph as JSON (load it back with Genome.from_dict):")
print(res.front[-1][0].to_dict())
