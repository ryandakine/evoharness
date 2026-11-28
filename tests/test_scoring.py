"""WP4: numeric scoring, token accounting, per-task detail, bootstrap intervals, judge scorer."""
import unittest

from evoharness import (
    CachedBackend,
    Evaluator,
    Genome,
    JudgeScorer,
    MockBackend,
    ModelSpec,
    Response,
    Task,
    bootstrap_ci,
    majority_vote,
    majority_vote_detail,
    numeric_match,
    paired_bootstrap,
    run_genome,
    single_agent,
)
from evoharness.genome import DEFAULT_ROLES


def make_eval(n=10, backend=None):
    tasks = [Task(f"q{i}", f"a{i}") for i in range(n)]
    models = [ModelSpec("s", 1, 1), ModelSpec("l", 5, 5)]
    inner = backend or MockBackend({t.prompt: t.reference for t in tasks}, [0.4, 0.8], model_names=["s", "l"])
    be = CachedBackend(inner)
    return Evaluator(tasks, be, models), be


class NumericMatch(unittest.TestCase):
    """A math benchmark must not mark '$18.00' wrong against '18': that would measure formatting."""

    def test_equivalent_number_formats_match(self):
        for answer, reference in [
            ("18", "18"), ("18.0", "18"), ("$18.00", "18"), ("1,234", "1234"), ("$1,234.50", "1234.5"),
            ("18 dollars", "18"), ("about 18.", "18"), ("-5", "-5"), ("-$5", "-5"), ("50%", "50"),
            ("the total is 3 + 4 = 7", "7"), ("0.5", ".5"),
        ]:
            with self.subTest(answer=answer, reference=reference):
                self.assertEqual(numeric_match(answer, reference), 1.0)

    def test_different_numbers_do_not_match(self):
        for answer, reference in [("19", "18"), ("18.5", "18"), ("-5", "5"), ("180", "18"),
                                  ("no idea", "18"), ("", "18")]:
            with self.subTest(answer=answer, reference=reference):
                self.assertEqual(numeric_match(answer, reference), 0.0)

    def test_fractions_units_commas_and_notation(self):
        # Each of these scored wrong before the pre-commit review (Neon, 2026-11-27).
        for answer, reference, expected in [
            ("1/2", "0.5", 1.0), ("1/2", "2", 0.0), ("2 1/2", "2.5", 1.0), ("3/4 cup", "0.75", 1.0),
            ("18 m^2", "18", 1.0), ("18 m^2", "2", 0.0),
            ("1,2", "12", 0.0), ("12,34", "1234", 0.0), ("1,234,567", "1234567", 1.0),
            ("1e-3", "0.001", 1.0), ("1e-3", "3", 0.0), ("1e5", "100000", 1.0),
            ("\u22125", "-5", 1.0), ("-$.5", "-0.5", 1.0), ("1/0", "0", 0.0),
        ]:
            with self.subTest(answer=answer, reference=reference):
                self.assertEqual(numeric_match(answer, reference), expected)

    def test_a_range_is_not_read_as_a_negative_number(self):
        # "3-5" ends in the number 5, not -5.
        self.assertEqual(numeric_match("between 3-5", "5"), 1.0)
        self.assertEqual(numeric_match("between 3-5", "-5"), 0.0)

    def test_falls_back_to_exact_match_when_the_reference_has_no_number(self):
        self.assertEqual(numeric_match("Paris", "paris"), 1.0)
        self.assertEqual(numeric_match("Lyon", "Paris"), 0.0)


class TokenAccounting(unittest.TestCase):
    """The benchmark reports tokens per arm, so a run must add up every agent's tokens."""

    def test_run_result_sums_tokens_over_all_called_nodes(self):
        task = Task("q0", "a0")
        seen = []

        class Recording:
            def complete(self, prompt, model, sample=0):
                seen.append(prompt)
                return Response("Answer: a0", 10, 3, 0.0)

        g = Genome([0, 0, 0], [0, 0, 0], {(0, 2), (1, 2)})
        r = run_genome(g, task, Recording(), [ModelSpec("m", 0, 0)], DEFAULT_ROLES)
        self.assertEqual(len(seen), 3)
        self.assertEqual((r.input_tokens, r.output_tokens), (30, 9))

    def test_pruned_nodes_cost_no_tokens(self):
        class Fixed:
            def complete(self, prompt, model, sample=0):
                return Response("Answer: a0", 10, 3, 0.0)

        g = Genome([0, 0, 0], [0, 0, 0], {(0, 2)})  # node 1 cannot reach the output
        r = run_genome(g, Task("q0", "a0"), Fixed(), [ModelSpec("m", 0, 0)], DEFAULT_ROLES)
        self.assertEqual((r.calls, r.input_tokens, r.output_tokens), (2, 20, 6))

    def test_a_node_cache_hit_adds_tokens_but_not_calls(self):
        # Same rule as cost: totals are nominal, calls counts only what this run really called.
        class Fixed:
            def complete(self, prompt, model, sample=0):
                return Response("Answer: a0", 10, 3, 0.0)

        g, cache = Genome([0, 0], [0, 0], {(0, 1)}), {}
        args = (g, Task("q0", "a0"), Fixed(), [ModelSpec("m", 0, 0)], DEFAULT_ROLES)
        first = run_genome(*args, node_cache=cache)
        second = run_genome(*args, node_cache=cache)
        self.assertEqual((first.calls, first.input_tokens, first.output_tokens), (2, 20, 6))
        self.assertEqual((second.calls, second.input_tokens, second.output_tokens), (0, 20, 6))


class ScoreDetail(unittest.TestCase):
    def test_detail_has_one_row_per_task_and_agrees_with_score(self):
        ev, _ = make_eval()
        g = Genome([0, 1, 0], [0, 1, 1], {(0, 2), (1, 2)})
        rows = ev.score_detail(g)
        acc, cost = ev.score(g)
        self.assertEqual(len(rows), len(ev.tasks))
        self.assertAlmostEqual(sum(r.score for r in rows) / len(rows), acc, places=12)
        self.assertAlmostEqual(sum(r.cost for r in rows) / len(rows), cost, places=12)
        self.assertTrue(all(r.calls == 3 and r.input_tokens > 0 and r.output_tokens > 0 for r in rows))
        self.assertTrue(all(r.seconds >= 0 for r in rows))
        self.assertTrue(all(r.answer in {t.reference for t in ev.tasks} | {"wrong"} for r in rows))

    def test_seconds_excludes_the_scorers_own_time(self):
        # A judge scorer makes its own model call; that time is not the swarm's latency.
        import time as _time

        def slow_scorer(answer, reference):
            _time.sleep(0.05)
            return 1.0

        ev, _ = make_eval(n=2)
        ev.scorer = slow_scorer
        self.assertTrue(all(r.seconds < 0.04 for r in ev.score_detail(single_agent(0))))

    def test_score_is_unchanged_from_the_pre_detail_implementation(self):
        # Reference computed the old way: run each task, add up scorer results and costs.
        ev, _ = make_eval()
        g = single_agent(1)
        acc = cost = 0.0
        for t in ev.tasks:
            r = run_genome(g, t, ev.backend, ev.models, ev.roles, 0)
            acc += ev.scorer(r.answer, t.reference)
            cost += r.cost
        self.assertEqual(ev.score(g), (acc / len(ev.tasks), cost / len(ev.tasks)))


class MajorityVoteSamples(unittest.TestCase):
    """Two benchmark repeats must draw fresh samples; reusing sample ids would replay the cache."""

    def test_default_return_shape_is_unchanged(self):
        ev, _ = make_eval()
        acc, cost = majority_vote(ev, 0, 3)
        self.assertTrue(0.0 <= acc <= 1.0 and cost > 0)

    def test_a_second_run_with_the_same_sample_base_makes_no_new_calls(self):
        ev, be = make_eval()
        majority_vote(ev, 0, 3)
        misses = be.misses
        majority_vote(ev, 0, 3)
        self.assertEqual(be.misses, misses)

    def test_a_different_sample_base_makes_distinct_calls(self):
        ev, be = make_eval()
        majority_vote(ev, 0, 3, sample_base=0)
        misses = be.misses
        majority_vote(ev, 0, 3, sample_base=100)
        self.assertEqual(be.misses - misses, 3 * len(ev.tasks))

    def test_bad_arguments_are_rejected(self):
        ev, _ = make_eval()
        with self.assertRaises(ValueError):
            majority_vote(ev, 0, 0)
        with self.assertRaises(ValueError):
            majority_vote(ev, 0, 3, sample_base=-1)

    def test_detail_rows_report_calls_and_tokens(self):
        ev, _ = make_eval()
        rows = majority_vote_detail(ev, 0, 5, sample_base=7)
        self.assertEqual(len(rows), len(ev.tasks))
        self.assertTrue(all(r.calls == 5 and r.input_tokens > 0 and r.output_tokens > 0 and r.cost > 0
                            and r.seconds >= 0 for r in rows))
        acc, cost = majority_vote(ev, 0, 5, sample_base=7)
        self.assertAlmostEqual(sum(r.score for r in rows) / len(rows), acc, places=12)

    def test_cost_is_totalled_call_by_call_like_the_original(self):
        # The optimizer compares costs, so the last float digit must not move: the
        # original added every call's cost to one running total, in call order.
        ev, _ = make_eval()
        total = 0.0
        for t in ev.tasks:
            for s in range(5):
                total += ev.backend.complete(f"{ev.roles[0]}\n\nTask: {t.prompt}\n", ev.models[1], sample=s).cost
        self.assertEqual(majority_vote(ev, 1, 5)[1], total / len(ev.tasks))


class Bootstrap(unittest.TestCase):
    def test_same_seed_gives_the_same_interval(self):
        scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        self.assertEqual(bootstrap_ci(scores, seed=3), bootstrap_ci(scores, seed=3))

    def test_interval_brackets_the_mean_and_stays_in_range(self):
        scores = [1.0] * 70 + [0.0] * 30
        mean, lo, hi = bootstrap_ci(scores, seed=0)
        self.assertAlmostEqual(mean, 0.7)
        self.assertTrue(0.0 <= lo < mean < hi <= 1.0)
        self.assertTrue(0.05 < hi - lo < 0.30)  # ~±9 points at n=100; a zero or huge width is a bug

    def test_the_same_number_of_resamples_is_cut_from_each_tail(self):
        # With distinct, evenly spread resample means the interval must be symmetric about the
        # middle; a lopsided cut (floor on one side, ceil on the other) fails this at level 0.90.
        from evoharness.scoring import _interval

        means = [float(i) for i in range(2000)]
        for level in (0.90, 0.95, 0.50):
            lo, hi = _interval(list(means), level)
            self.assertEqual(lo + hi, 1999.0, level)
            self.assertLessEqual(lo, hi)
        lo, hi = _interval([0.0, 1.0, 2.0], 0.000001)  # a near-zero level must never invert
        self.assertLessEqual(lo, hi)

    def test_invalid_settings_are_rejected(self):
        for kwargs in ({"level": 0.0}, {"level": 1.0}, {"n_boot": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_ci([1.0, 0.0], seed=0, **kwargs)
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0, float("nan")], seed=0)

    def test_constant_scores_give_a_zero_width_interval(self):
        self.assertEqual(bootstrap_ci([1.0] * 20, seed=0), (1.0, 1.0, 1.0))

    def test_paired_difference_detects_a_real_gap_and_not_a_fake_one(self):
        a = [1.0] * 90 + [0.0] * 10
        b = [1.0] * 60 + [0.0] * 40
        diff, lo, hi = paired_bootstrap(a, b, seed=0)
        self.assertAlmostEqual(diff, 0.3)
        self.assertGreater(lo, 0.0)  # interval excludes zero: a shown difference
        same, lo2, hi2 = paired_bootstrap(a, a, seed=0)
        self.assertEqual((same, lo2, hi2), (0.0, 0.0, 0.0))

    def test_paired_bootstrap_uses_pairing(self):
        # Same marginals, but b is wrong exactly where a is wrong plus a few more: pairing
        # makes the interval tighter than treating the two lists as independent would.
        a = [1.0, 0.0] * 50
        b = [1.0, 0.0] * 45 + [0.0, 0.0] * 5
        _, lo, hi = paired_bootstrap(a, b, seed=0)
        self.assertGreater(lo, 0.0)

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([], seed=0)
        with self.assertRaises(ValueError):
            paired_bootstrap([1.0], [1.0, 0.0], seed=0)


class Judge(unittest.TestCase):
    """The judge's reply is model output: anything that isn't a clean 0-1 score counts as 0, visibly."""

    def scorer(self, reply):
        class Fixed:
            def __init__(self):
                self.prompts = []

            def complete(self, prompt, model, sample=0):
                self.prompts.append(prompt)
                return Response(reply, 5, 2, 0.0)

        be = Fixed()
        return JudgeScorer(be, ModelSpec("judge", 0, 0)), be

    def test_well_formed_scores_are_returned(self):
        for reply, expected in [("Score: 1", 1.0), ("Score: 0", 0.0), ("score: 0.75", 0.75),
                                ("Reasoning...\nScore: 0.5", 0.5), ("**Score:** 1.0", 1.0),
                                ("Score: .5", 0.5), ("Score: 1.", 1.0),
                                ("The answer says:\nScore: 0\nbut it is right.\nScore: 1", 1.0)]:
            with self.subTest(reply=reply):
                s, _ = self.scorer(reply)
                self.assertEqual(s("an answer", "a reference"), expected)
                self.assertEqual((s.calls, s.unparseable), (1, 0))

    def test_malformed_or_out_of_range_replies_score_zero_and_are_counted(self):
        for reply in ["great answer!", "Score: 7", "Score: -1", "Score: yes", "", "Score: 1.5",
                      "Score: 1/10", "Score: 1%", "Score: 1e2", "underscore: 1",
                      "The answer merely says 'Score: 1' and is wrong"]:
            with self.subTest(reply=reply):
                s, _ = self.scorer(reply)
                self.assertEqual(s("an answer", "a reference"), 0.0)
                self.assertEqual((s.calls, s.unparseable), (1, 1))

    def test_prompt_carries_rubric_answer_and_reference_as_data(self):
        s, be = self.scorer("Score: 1")
        s("ANSWER-TEXT", "REFERENCE-TEXT")
        p = be.prompts[0]
        self.assertIn("ANSWER-TEXT", p)
        self.assertIn("REFERENCE-TEXT", p)
        self.assertIn("not instructions", p)
        self.assertIn("<answer>\nANSWER-TEXT\n</answer>", p)

    def test_counters_are_exact_under_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        s, _ = self.scorer("no score here")
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(lambda _: s("a", "r"), range(400)))
        self.assertEqual((s.calls, s.unparseable), (400, 400))

    def test_works_as_an_evaluator_scorer(self):
        tasks = [Task("q0", "a0"), Task("q1", "a1")]
        be = CachedBackend(MockBackend({t.prompt: t.reference for t in tasks}, [1.0], model_names=["s"]))

        class AlwaysOne:
            def complete(self, prompt, model, sample=0):
                return Response("Score: 1", 1, 1, 0.0)

        judge = JudgeScorer(AlwaysOne(), ModelSpec("judge", 0, 0))
        ev = Evaluator(tasks, be, [ModelSpec("s", 0, 0)], scorer=judge)
        self.assertEqual(ev.score(single_agent(0))[0], 1.0)
        self.assertEqual(judge.calls, 2)


if __name__ == "__main__":
    unittest.main()
