import random
import unittest

from evoharness import (
    BudgetExceeded,
    CachedBackend,
    Evaluator,
    GAConfig,
    Genome,
    MockBackend,
    ModelSpec,
    Task,
    evolve,
    single_agent,
)
from evoharness.ga import dominates, nondominated_sort
from evoharness.genome import crossover, mutate, random_genome


def make_eval(budget=None):
    tasks = [Task(f"q{i}", f"a{i}") for i in range(10)]
    models = [ModelSpec("s", 1, 1), ModelSpec("l", 5, 5)]
    be = CachedBackend(MockBackend({t.prompt: t.reference for t in tasks}, [0.4, 0.8]), budget)
    return Evaluator(tasks, be, models), be


class Core(unittest.TestCase):
    def test_pruning_drops_disconnected_nodes(self):
        self.assertEqual(Genome([0, 0, 0], [0, 0, 0], {(0, 2)}).active_nodes(), [0, 2])

    def test_operators_keep_dag_valid(self):
        rng = random.Random(1)
        a, b = (random_genome(rng, 5, 3, 2) for _ in range(2))
        for g in (mutate(a, rng, 3, 2, 0.5), crossover(a, b, rng)):
            self.assertTrue(all(i < j for i, j in g.edges))

    def test_nondominated_sort(self):
        objs = [(0.1, 5), (0.2, 3), (0.3, 4), (0.1, 6)]
        self.assertEqual(set(nondominated_sort(objs)[0]), {0, 1})
        self.assertTrue(dominates(objs[0], objs[3]))

    def test_ga_front_nondominated_and_deterministic(self):
        ev, _ = make_eval()
        cfg = GAConfig(n_nodes=3, pop_size=8, generations=3, seed=7)
        r1 = evolve(cfg, ev.objectives)
        r2 = evolve(cfg, ev.objectives)
        objs = [o for _, o in r1.front]
        self.assertFalse(any(dominates(a, b) for a in objs for b in objs))
        self.assertEqual(objs, [o for _, o in r2.front])

    def test_cache_hits_are_free(self):
        ev, be = make_eval()
        ev.score(single_agent(0))
        misses = be.misses
        ev.score(single_agent(0))
        self.assertEqual(be.misses, misses)
        self.assertGreater(be.hits, 0)

    def test_budget_cap_raises(self):
        ev, be = make_eval(budget=1e-9)
        m = ev.models[0]
        be.complete("first", m)            # allowed: nothing spent yet
        with self.assertRaises(BudgetExceeded):
            be.complete("second", m)       # cap now exceeded


if __name__ == "__main__":
    unittest.main()
