import random
import unittest

from evoharness import (
    Attribution,
    DistributionConfig,
    Evaluator,
    Experience,
    Genome,
    ModelSpec,
    ParticleSwarm,
    PSOConfig,
    Response,
    Task,
    dag_shapley,
    evaluate_frozen,
    g_decode,
    jfk_scores,
    optimize_prompts,
    reinforce_graph,
    run_genome,
    search_roles,
    shapley,
    split_tasks,
    viable,
)


class DeterministicBackend:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, model, sample=0):
        self.calls.append((prompt, model, sample))
        # Include input in output so distinct upstream configurations remain
        # distinct, rather than accidentally collapsing to a constant cache key.
        return Response(prompt + "\nAnswer: yes", 1, 1, 0.01)


def layered():
    return Genome([0] * 7, [0] * 7,
                  {(i, j) for i in range(3) for j in range(3, 6)} |
                  {(j, 6) for j in range(3, 6)})


class AttributionTests(unittest.TestCase):
    def test_disconnected_dummy_and_exactness_limit(self):
        g = Genome([0, 0, 0], [0, 0, 0], {(0, 2)})
        result = dag_shapley(g, lambda _: 1.0)
        self.assertAlmostEqual(result.scores[1], 0.0)
        self.assertAlmostEqual(result.scores[0], 0.5)
        self.assertAlmostEqual(result.scores[2], 0.5)
        with self.assertRaises(ValueError):
            dag_shapley(g, lambda _: 1.0, max_nodes=2)

    def test_additive_game_and_efficiency(self):
        weights = [1, -2, 3, 0]
        a = shapley(4, lambda s: 5 + sum(weights[i] for i in s))
        for actual, expected in zip(a.scores, weights):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(a.scores), a.values[frozenset(range(4))] - a.values[frozenset()])

    def test_paper_coalitions_and_full_shapley_equivalence(self):
        g = layered()
        def callback(s):
            return sum(i + 1 for i in s)
        fast = dag_shapley(g, callback)
        slow = shapley(g.n, lambda s: callback(s) if viable(g, s) else 0)
        self.assertEqual(fast.evaluated, 49)
        self.assertEqual(fast.pruned, 79)
        self.assertEqual(fast.scores, slow.scores)
        self.assertAlmostEqual(sum(fast.scores), callback(set(range(7))))

    def test_missing_original_source_is_not_a_new_source(self):
        g = Genome([0] * 3, [0] * 3, {(0, 1), (1, 2)})
        self.assertFalse(viable(g, {1, 2}))
        self.assertFalse(viable(g, {0, 2}))
        self.assertTrue(viable(g, {0, 1, 2}))

    def test_hierarchical_memoization_paper_call_count(self):
        g = layered()
        backend = DeterministicBackend()
        ev = Evaluator([Task('question', 'yes')], backend, [ModelSpec('m', 1, 1)])
        result = ev.attribute(g, deterministic=True)
        self.assertEqual(len(backend.calls), 73)
        backend.calls.clear()
        uncached = ev.attribute(g, deterministic=True, memoize=False)
        self.assertGreater(len(backend.calls), 73)
        self.assertEqual(result.values, uncached.values)
        self.assertEqual(result.scores, uncached.scores)
        with self.assertRaises(ValueError):
            ev.attribute(g)

    def test_external_data_access(self):
        g = Genome([0, 0], [0, 0], {(0, 1)})
        backend = DeterministicBackend()
        run_genome(g, Task('secret raw data', 'yes'), backend, [ModelSpec('m', 1, 1)], ['role'],
                   external_inputs={0: 'source data'})
        self.assertIn('Task: source data', backend.calls[0][0])
        self.assertIn('Task: \n', backend.calls[1][0])
        self.assertNotIn('secret raw data', backend.calls[1][0])

    def test_prompt_changes_do_not_reuse_old_node_outputs(self):
        g = Genome([0, 0], [0, 0], {(0, 1)})
        backend = DeterministicBackend()
        ev = Evaluator([Task('question', 'yes')], backend, [ModelSpec('m', 0, 0)], ['old'])
        ev.attribute(g, deterministic=True)
        backend.calls.clear()
        ev.roles = ['new']
        ev.attribute(g, deterministic=True)
        self.assertTrue(backend.calls)
        self.assertTrue(all(prompt.startswith('new') for prompt, _, _ in backend.calls))


class OptimizationTests(unittest.TestCase):
    def test_reinforce_learns_rewarding_edge(self):
        g = Genome([0, 0], [0, 0])
        cfg = DistributionConfig(iterations=150, batch_size=32, learning_rate=0.5, seed=4)
        result = reinforce_graph(g, lambda x: float((0, 1) in x.edges), cfg)
        self.assertGreater(result.distribution.probabilities[0], 0.95)
        self.assertEqual(result.best_utility, 1)
        self.assertEqual(result.history, reinforce_graph(g, lambda x: float((0, 1) in x.edges), cfg).history)

    def test_pso_numeric_update_and_snapshot(self):
        cfg = PSOConfig(inertia=0, cognitive=0, social=1, repel=0, seed=3)
        s = ParticleSwarm([[0.0], [2.0]], cfg)
        s.observe([0.0, 1.0])
        s.move()
        self.assertEqual(s.positions, [[2.0], [2.0]])
        s.positions[1][0] = 100
        self.assertEqual(s.best, [2.0])

    def test_pso_repulsion_direction(self):
        s = ParticleSwarm([[0.0], [2.0]], PSOConfig(inertia=0, cognitive=0, social=0, repel=1))
        s.observe([0.0, 1.0])
        s.move()
        self.assertEqual(s.positions, [[0.0], [4.0]])

    def test_decode_preserves_identities_and_varies_sink(self):
        g = Genome([0, 1, 2], [0, 1, 2])
        matrix = [[1.0] * 3 for _ in range(3)]
        sinks = set()
        for seed in range(30):
            decoded = g_decode(matrix, g, random.Random(seed), top_p=1)
            self.assertEqual(sorted(zip(decoded.roles, decoded.models)), [(0, 0), (1, 1), (2, 2)])
            self.assertTrue(all(i < j for i, j in decoded.edges))
            sinks.add(decoded.models[-1])
        self.assertEqual(sinks, {0, 1, 2})

    def test_role_search_returns_measured_graph(self):
        cfg = PSOConfig(particles=4, iterations=4, seed=1)
        def utility(g):
            return float(len(g.edges))
        r = search_roles(Genome([0] * 4, [0] * 4), utility, cfg)
        self.assertEqual(r.utility, utility(r.genome))
        self.assertEqual(r.genome.key(), search_roles(Genome([0] * 4, [0] * 4), utility, cfg).genome.key())

    def test_jfk_frequency_weighting(self):
        g, seen = Genome([0, 0, 0], [0, 0, 0]), []
        def utility(graph):
            seen.append(list(graph.models))
            return float(sum(graph.models))
        scores = jfk_scores(g, 2, utility, assignments=8, seed=9)
        for i in range(2):
            counts = [assignment.count(i) for assignment in seen]
            self.assertAlmostEqual(scores[i], sum(c * sum(a) for c, a in zip(counts, seen)) / sum(counts))

    def test_reflection_changes_only_bottleneck_even_with_shared_role(self):
        g, roles = Genome([0, 0], [0, 0], {(0, 1)}), ['Original instruction']
        a = Attribution([-1, 1], {}, 0, 0)
        history = [Experience(0, 'failure', 'bad answer', -1), Experience(0, 'success', 'good answer', 1)]
        backend = DeterministicBackend()
        updated = optimize_prompts(g, roles, history, lambda _: a, backend, ModelSpec('reflector', 0, 0))
        self.assertEqual(updated.target, 0)
        self.assertEqual(updated.roles[updated.genome.roles[1]], roles[0])
        self.assertIn('Lessons from experience', updated.roles[updated.genome.roles[0]])
        self.assertEqual(g.roles, [0, 0])
        self.assertEqual(roles, ['Original instruction'])
        self.assertIn('failures', backend.calls[0][0])
        self.assertIn('successes', backend.calls[0][0])
        backend.calls.clear()
        self.assertIsNone(optimize_prompts(g, roles, history, lambda _: a, backend,
                                          ModelSpec('r', 0, 0), threshold=-2).target)
        self.assertEqual(backend.calls, [])

    def test_reflection_skips_without_target_history(self):
        g = Genome([0], [0])
        backend = DeterministicBackend()
        result = optimize_prompts(g, ['role'], [], lambda _: Attribution([-1], {}, 0, 0),
                                  backend, ModelSpec('r', 0, 0))
        self.assertIsNone(result.target)
        self.assertEqual(backend.calls, [])


class EvaluationTests(unittest.TestCase):
    def test_split_groups_duplicate_prompts(self):
        tasks = [Task(str(i), 'yes') for i in range(10)] + [Task('0', 'yes')]
        split = split_tasks(tasks)
        sets = [{t.prompt for t in part} for part in (split.train, split.validation, split.test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, (split.train, split.validation, split.test))), len(tasks))

    def test_chronological_split_and_leakage_guard(self):
        tasks = [Task(str(i), 'yes') for i in range(10)]
        split = split_tasks(tasks, chronological=True)
        self.assertEqual([t.prompt for t in split.test], ['8', '9'])
        ev = Evaluator(split.test, DeterministicBackend(), [ModelSpec('m', 0, 0)])
        g = Genome([0], [0])
        result = evaluate_frozen([g], ev, search_tasks=split.train + split.validation, samples=3)
        self.assertEqual(result[0].sample_accuracies, [1, 1, 1])
        with self.assertRaises(ValueError):
            evaluate_frozen([g], ev, search_tasks=tasks)

    def test_invalid_graph_and_empty_evaluator(self):
        for roles, models, edges in [([], [], set()), ([0], [0], {(0, 0)}), ([0], [], set())]:
            with self.assertRaises(ValueError):
                Genome(roles, models, edges)
        with self.assertRaises(ValueError):
            Evaluator([], DeterministicBackend(), [ModelSpec('m', 0, 0)])


if __name__ == '__main__':
    unittest.main()
