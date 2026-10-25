"""Coverage for the 0.2 improvements: answer parsing, serialization, validation, parallelism, disk cache."""
import json
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path

from evoharness import (
    AgentError,
    BudgetExceeded,
    CachedBackend,
    Evaluator,
    GAConfig,
    Genome,
    MockBackend,
    ModelSpec,
    Response,
    Task,
    evolve,
    extract_answer,
    run_genome,
)
from evoharness.genome import crossover


class AnswerParsing(unittest.TestCase):
    def test_answer_line_variants(self):
        cases = {
            "Working...\nAnswer: 42": "42",
            "**Answer:** 42": "42",
            "**Answer**: 42": "42",
            "Final Answer: Paris.": "Paris.",
            "final answer: paris": "paris",
            "Answer:\n42": "42",
            "Answer: 1\nAnswer: 2": "2",
            "Therefore, Answer: `x = 3`": "x = 3",
            "- **Final Answer:** 7": "7",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_answer(text), expected)

    def test_boxed_and_fallbacks(self):
        self.assertEqual(extract_answer(r"so we get $\boxed{\frac{1}{2}}$ done"), r"\frac{1}{2}")
        self.assertEqual(extract_answer("No format here\n\nJust the last line  \n\n"), "Just the last line")
        self.assertEqual(extract_answer("   \n  "), "")
        self.assertEqual(extract_answer("Answer: 5\nbut \\boxed{6}"), "5")


class Serialization(unittest.TestCase):
    def test_round_trip(self):
        g = Genome([0, 1, 2], [0, 0, 1], {(0, 2), (1, 2)})
        data = json.loads(json.dumps(g.to_dict()))
        self.assertEqual(data["edges"], [[0, 2], [1, 2]])
        self.assertEqual(Genome.from_dict(data).key(), g.key())

    def test_bad_dicts(self):
        for bad in ({}, {"roles": [0], "models": [0], "edges": [[0]]}, {"roles": [0], "models": [0], "edges": "x"},
                    {"roles": [0], "models": [0, 1]}, {"roles": ["a"], "models": [0]}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Genome.from_dict(bad)


class Validation(unittest.TestCase):
    def test_crossover_and_seed_size_mismatch(self):
        rng = random.Random(0)
        with self.assertRaises(ValueError):
            crossover(Genome([0], [0]), Genome([0, 0], [0, 0]), rng)
        with self.assertRaises(ValueError):
            evolve(GAConfig(n_nodes=3, generations=1), lambda g: (0.0, 0.0), seeds=[Genome([0], [0])])
        with self.assertRaises(ValueError):
            evolve(GAConfig(n_nodes=1, n_roles=2, generations=1), lambda g: (0.0, 0.0), seeds=[Genome([5], [0])])
        with self.assertRaises(ValueError):
            evolve(GAConfig(pop_size=1), lambda g: (0.0, 0.0))

    def test_seeds_are_not_mutated_by_evolve(self):
        seed = Genome([0, 0, 0], [0, 0, 0], {(0, 2)})
        evolve(GAConfig(n_nodes=3, pop_size=4, generations=2), lambda g: (float(len(g.edges)), 0.0), seeds=[seed])
        self.assertEqual(seed.key(), Genome([0, 0, 0], [0, 0, 0], {(0, 2)}).key())

    def test_run_genome_rejects_out_of_range_indices(self):
        backend = MockBackend({"q": "a"}, [0.9])
        with self.assertRaisesRegex(ValueError, "role index"):
            run_genome(Genome([3], [0]), Task("q", "a"), backend, [ModelSpec("m", 0, 0)], ["only role"])
        with self.assertRaisesRegex(ValueError, "model index"):
            run_genome(Genome([0], [1]), Task("q", "a"), backend, [ModelSpec("m", 0, 0)], ["only role"])
        with self.assertRaises(ValueError):
            run_genome(Genome([0], [0]), Task("q", "a"), backend, [ModelSpec("m", 0, 0)], ["r"], max_workers=0)


class FailingBackend:
    def __init__(self, fail_node_prompt_marker, exc):
        self.marker, self.exc = fail_node_prompt_marker, exc

    def complete(self, prompt, model, sample=0):
        if self.marker in prompt:
            raise self.exc
        return Response("Answer: ok", 1, 1, 0.0)


class Errors(unittest.TestCase):
    def test_agent_error_names_the_node(self):
        g = Genome([0, 1], [0, 0], {(0, 1)})
        backend = FailingBackend("verifier", RuntimeError("boom"))
        with self.assertRaises(AgentError) as ctx:
            run_genome(g, Task("q", "a"), backend, [ModelSpec("m", 0, 0)], ["solver", "verifier"])
        self.assertEqual(ctx.exception.node, 1)
        self.assertEqual(ctx.exception.model_name, "m")
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertIn("agent 1 (m) failed: boom", str(ctx.exception))

    def test_budget_exceeded_passes_through_unwrapped(self):
        g = Genome([0], [0])
        backend = FailingBackend("solver", BudgetExceeded("cap"))
        with self.assertRaises(BudgetExceeded):
            run_genome(g, Task("q", "a"), backend, [ModelSpec("m", 0, 0)], ["solver"])


class SlowRecordingBackend:
    """Sleeps per call and records how many calls overlapped in time."""

    def __init__(self, delay=0.1):
        self.delay, self.active, self.peak, self.calls = delay, 0, 0, []
        self.lock = threading.Lock()

    def complete(self, prompt, model, sample=0):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append(prompt)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return Response(prompt + "\nAnswer: yes", 1, 1, 0.01)


class Parallelism(unittest.TestCase):
    def layered(self):
        return Genome([0] * 7, [0] * 7, {(i, j) for i in range(3) for j in range(3, 6)} | {(j, 6) for j in range(3, 6)})

    def test_parallel_matches_sequential_and_overlaps_independent_nodes(self):
        g, task, models, roles = self.layered(), Task("q", "yes"), [ModelSpec("m", 1, 1)], ["role"]
        seq_backend, par_backend = SlowRecordingBackend(), SlowRecordingBackend()
        seq = run_genome(g, task, seq_backend, models, roles)
        par = run_genome(g, task, par_backend, models, roles, max_workers=3)
        self.assertEqual(seq.outputs, par.outputs)
        self.assertEqual(list(par.outputs), list(range(7)))
        self.assertEqual((seq.answer, seq.calls, seq.cost), (par.answer, par.calls, par.cost))
        self.assertEqual(seq_backend.peak, 1)
        self.assertEqual(par_backend.peak, 3)
        self.assertEqual(sorted(seq_backend.calls), sorted(par_backend.calls))

    def test_parallel_respects_coalitions_and_node_cache(self):
        g, task, models, roles = self.layered(), Task("q", "yes"), [ModelSpec("m", 1, 1)], ["role"]
        cache = {}
        first = run_genome(g, task, SlowRecordingBackend(0), models, roles, coalition={0, 3, 6},
                           node_cache=cache, max_workers=4)
        again = run_genome(g, task, SlowRecordingBackend(0), models, roles, coalition={0, 3, 6},
                           node_cache=cache, max_workers=4)
        self.assertEqual(first.calls, 3)
        self.assertEqual(again.calls, 0)
        self.assertEqual(first.outputs, again.outputs)
        self.assertEqual(run_genome(g, task, SlowRecordingBackend(0), models, roles, coalition={0, 3}).calls, 0)

    def test_parallel_evaluator_with_mock_backend_is_deterministic(self):
        tasks = [Task(f"q{i}", f"a{i}") for i in range(12)]
        models = [ModelSpec("s", 1, 1), ModelSpec("l", 5, 5)]
        g = Genome([0, 1, 2, 0], [0, 1, 0, 1], {(0, 3), (1, 3), (2, 3)})
        results = set()
        for workers in (1, 3, 3, 8):
            be = CachedBackend(MockBackend({t.prompt: t.reference for t in tasks}, [0.4, 0.8], model_names=["s", "l"]))
            results.add(Evaluator(tasks, be, models, max_workers=workers).score(g))
        self.assertEqual(len(results), 1)
        with self.assertRaises(ValueError):
            Evaluator(tasks, MockBackend({}, [0.5]), models, max_workers=0)


class DiskCache(unittest.TestCase):
    def test_persists_across_instances_and_tolerates_junk_lines(self):
        model = ModelSpec("m", 1, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "cache.jsonl"
            inner = SlowRecordingBackend(0)
            first = CachedBackend(inner, cache_path=path)
            r1 = first.complete("hello", model)
            first.complete("hello", model, sample=1)
            self.assertEqual((first.hits, first.misses, len(first)), (0, 2, 2))
            with path.open("a", encoding="utf-8") as fh:
                fh.write("not json\n{\"key\": \"partial\"}\n")
            second = CachedBackend(SlowRecordingBackend(0), cache_path=path)
            self.assertEqual(len(second), 2)
            self.assertEqual(second.complete("hello", model), r1)
            self.assertEqual((second.hits, second.misses), (1, 0))
            repriced = second.complete("hello", ModelSpec("m", 10, 20))
            self.assertEqual(repriced.text, r1.text)
            self.assertAlmostEqual(repriced.cost, (1 * 10 + 1 * 20) / 1e6)
            self.assertEqual(second.hits, 2)

    def test_budget_counts_only_real_spend(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.jsonl"
            CachedBackend(SlowRecordingBackend(0), cache_path=path).complete("p", ModelSpec("m", 1, 1))
            reopened = CachedBackend(SlowRecordingBackend(0), budget_usd=0.005, cache_path=path)
            reopened.complete("p", ModelSpec("m", 1, 1))  # hit: free
            reopened.complete("new", ModelSpec("m", 1, 1))  # first real call allowed
            with self.assertRaises(BudgetExceeded):
                reopened.complete("newer", ModelSpec("m", 1, 1))


if __name__ == "__main__":
    unittest.main()
