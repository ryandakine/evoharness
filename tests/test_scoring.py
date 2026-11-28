"""WP4: numeric scoring, token accounting, per-task detail, bootstrap intervals, judge scorer."""
import itertools
import unittest
from fractions import Fraction

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
    canonical_number,
    exact_match,
    majority_vote,
    majority_vote_detail,
    numeric_key,
    numeric_match,
    paired_bootstrap,
    run_genome,
    single_agent,
)
from evoharness.genome import DEFAULT_ROLES
from evoharness.tools.base import ExecutionContext, Tool, ToolRegistry, ToolResult


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

    def test_integer_reference_needs_exact_integer(self):
        for answer, reference, expected in [
            ("1000001", "1000000", 0.0), ("5000004", "5000000", 0.0), ("999999", "1000000", 0.0),
            ("123456789", "123456788", 0.0), ("1000000.5", "1000000", 0.0),
            # D=6: anything that rounds to the same 6 places is one answer (was: abs 1e-9 for integers).
            ("18.0000001", "18", 1.0), ("18.0000004", "18", 1.0), ("18.000001", "18", 0.0),
            ("-1000001", "-1000000", 0.0), ("1,000,001", "1,000,000", 0.0),
            ("1000000", "1,000,000", 1.0), ("1e6", "1000000", 1.0), ("0.333333", "1/3", 1.0), ("18.0", "18", 1.0),
        ]:
            with self.subTest(answer=answer, reference=reference):
                self.assertEqual(numeric_match(answer, reference), expected)

    def test_tolerance_for_decimal_references_is_capped_in_absolute_terms(self):
        self.assertEqual(numeric_match("1000001.5", "1000000.5"), 0.0)
        self.assertEqual(numeric_match("1000000.5", "1000000.4"), 0.0)
        self.assertEqual(numeric_match("1000000.5", "1000000.5"), 1.0)

    def test_infinite_reference_does_not_crash(self):
        # "1e999" parses to inf; int(inf) would raise OverflowError.
        for answer, reference in [("5", "1e999"), ("1e999", "1e999"), ("1e999", "5")]:
            with self.subTest(answer=answer, reference=reference):
                self.assertIn(numeric_match(answer, reference), (0.0, 1.0))
        self.assertEqual(numeric_match("5", "1e999"), 0.0)
        numeric_key("1e999")

    def test_a_range_is_not_read_as_a_negative_number(self):
        # "3-5" ends in the number 5, not -5.
        self.assertEqual(numeric_match("between 3-5", "5"), 1.0)
        self.assertEqual(numeric_match("between 3-5", "-5"), 0.0)

    def test_no_number_never_matches_so_text_benchmarks_use_exact_match(self):
        # Was: a reference without a number fell back to exact_match. Now a non-number is None
        # and None never matches, which also stops "" matching "$".
        for answer, reference in [("Paris", "paris"), ("Paris", "Paris"), ("Lyon", "Paris"), ("", "$"), ("", "")]:
            with self.subTest(answer=answer, reference=reference):
                self.assertEqual(numeric_match(answer, reference), 0.0)
        self.assertEqual(exact_match("Paris", "paris"), 1.0)

    def test_decimal_policy_is_a_six_place_bucket_not_a_relative_tolerance(self):
        # Honest D=6 policy: 0.1 now matches 0.1000004 (old tolerance there was 1e-7), and
        # +/-0.0000004 share the zero bucket (old: 0.0). Buckets split at half-units.
        for answer, reference, expected in [
            ("0.1", "0.1000004", 1.0), ("0.1", "0.1000006", 0.0), ("0.0000004", "-0.0000004", 1.0),
            ("0.0000004", "0", 1.0), ("-0.0000004", "0", 1.0), ("0.0000006", "0", 0.0),
            ("0.3333334", "0.3333335", 0.0), ("0.3333335", "0.3333336", 1.0), ("1/3", "0.3333335", 0.0),
            ("1/3", "0.333333", 1.0),
        ]:
            with self.subTest(answer=answer, reference=reference):
                self.assertEqual(numeric_match(answer, reference), expected)
                self.assertIs(type(numeric_match(answer, reference)), float)

    def test_integers_beyond_2_pow_53_are_exact(self):
        self.assertEqual(numeric_match("9007199254740993", "9007199254740992"), 0.0)  # old: 1.0
        self.assertEqual(numeric_match("9007199254740993", "9007199254740993"), 1.0)
        self.assertEqual(numeric_match("9007199254740993.0", "9007199254740993"), 1.0)

    def test_supported_grammar_pins_known_defects(self):
        """Pins what the unchanged _NUMBER regex does today. The first four are parser defects
        (Taskmaster task 17), pinned so a fix is a deliberate edit of this test, not a surprise."""
        for text, expected in [
            (".5e-3", 3),                  # defect: should be 0.0005
            (".5e3", Fraction(1, 2)),      # defect: should be 500
            ("1/-2", -2),                  # defect: not a fraction, last token is -2
            ("1,234 1/2", Fraction(1, 2)), # defect: should be 1234.5; the fraction wins
            ("1,2", 2), ("12,34", 34),     # a bad comma group splits into two numbers; the last wins
            ("50%", 50), ("3:30", 30), ("$18 for 3 days", 3), ("between 3-5", 5), ("18 m^2", 18),
        ]:
            with self.subTest(text=text):
                self.assertEqual(canonical_number(text), expected)

    def test_signed_mixed_fractions(self):
        for text, expected in [("2 1/2", Fraction(5, 2)), ("-2 1/2", Fraction(-5, 2)),
                               ("-0 1/2", Fraction(-1, 2)), ("0 1/2", Fraction(1, 2)),
                               ("\u22122 1/2", Fraction(-5, 2))]:
            with self.subTest(text=text):
                self.assertEqual(canonical_number(text), expected)
        self.assertEqual(numeric_match("-2 1/2", "-2.5"), 1.0)
        self.assertEqual(numeric_match("-2 1/2", "2.5"), 0.0)


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


    def test_react_node_sums_tokens_over_all_turns(self):
        class Echo(Tool):
            name, description, parameters = "echo", "echo", {"type": "object", "properties": {}}

            def execute(self, args, context):
                return ToolResult("ok")

        class Turns:
            def __init__(self):
                self.n = 0

            def complete(self, prompt, model, sample=0):
                raise AssertionError("fell back to complete(): the ReAct path was not taken")

            def complete_messages(self, messages, model, sample=0, tools=None, tool_choice=None):
                self.n += 1
                if self.n == 1:
                    call = {"id": "1", "function": {"name": "echo", "arguments": "{}"}}
                    return Response("", 10, 3, 0.0, tool_calls=[call])
                return Response("Answer: a0", 7, 2, 0.0)

        reg = ToolRegistry()
        reg.register(Echo())
        r = run_genome(Genome([0], [0]), Task("q0", "a0"), Turns(), [ModelSpec("m", 0, 0)], DEFAULT_ROLES,
                       tool_registry=reg, context=ExecutionContext())
        self.assertEqual(r.tool_calls, 1)  # proves the loop ran, not a silent fallback
        self.assertEqual((r.calls, r.input_tokens, r.output_tokens), (1, 17, 5))

    def test_react_synthesis_turn_tokens_are_counted(self):
        class Echo(Tool):
            name, description, parameters = "echo", "echo", {"type": "object", "properties": {}}

            def execute(self, args, context):
                return ToolResult("ok")

        class Greedy:
            def complete(self, prompt, model, sample=0):
                raise AssertionError("fell back to complete(): the ReAct path was not taken")

            def complete_messages(self, messages, model, sample=0, tools=None, tool_choice=None):
                if tool_choice == "none":
                    return Response("Answer: a0", 4, 1, 0.0)
                call = {"id": "1", "function": {"name": "echo", "arguments": "{}"}}
                return Response("", 10, 3, 0.0, tool_calls=[call])

        reg = ToolRegistry()
        reg.register(Echo())
        r = run_genome(Genome([0], [0]), Task("q0", "a0"), Greedy(), [ModelSpec("m", 0, 0)], DEFAULT_ROLES,
                       tool_registry=reg, context=ExecutionContext(), max_tool_turns=1)
        self.assertEqual(r.answer, "a0")  # the synthesis reply, so that call really happened
        self.assertEqual((r.input_tokens, r.output_tokens), (14, 4))

    def test_parallel_run_sums_tokens_like_sequential(self):
        class Fixed:
            def complete(self, prompt, model, sample=0):
                return Response("Answer: a0", 10, 3, 0.0)

        g = Genome([0, 0, 0, 0], [0, 0, 0, 0], {(0, 3), (1, 3), (2, 3)})
        for workers in (1, 3):
            with self.subTest(max_workers=workers):
                r = run_genome(g, Task("q0", "a0"), Fixed(), [ModelSpec("m", 0, 0)], DEFAULT_ROLES,
                               max_workers=workers)
                self.assertEqual((r.calls, r.input_tokens, r.output_tokens), (4, 40, 12))


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

    def test_vote_seconds_excludes_the_scorers_own_time(self):
        import time as _time

        def slow_scorer(answer, reference):
            _time.sleep(0.2)
            return 1.0

        ev, _ = make_eval(n=2)
        ev.scorer = slow_scorer
        self.assertTrue(all(r.seconds < 0.1 for r in majority_vote_detail(ev, 0, 1)))

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
        for t, r in zip(ev.tasks, rows):
            per_call = [ev.backend.complete(f"{ev.roles[0]}\n\nTask: {t.prompt}\n", ev.models[0], sample=s)
                        for s in range(7, 12)]
            self.assertEqual(r.calls, 5)
            self.assertEqual(r.input_tokens, sum(c.input_tokens for c in per_call))
            self.assertEqual(r.output_tokens, sum(c.output_tokens for c in per_call))
            self.assertTrue(r.input_tokens > 0 and r.output_tokens > 0 and r.cost > 0 and r.seconds >= 0)
            self.assertIn(r.answer, {t.reference, "wrong"})
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


class Scripted:
    """Backend whose reply for sample s is replies[s]."""

    def __init__(self, replies):
        self.replies = replies

    def complete(self, prompt, model, sample=0):
        return Response(self.replies[sample], 1, 1, 0.0)


def scripted_eval(*replies, ref="18"):
    return Evaluator([Task("q", ref)], Scripted(list(replies)), [ModelSpec("s", 1, 1)], scorer=numeric_match)


class NumericVote(unittest.TestCase):
    """The scorer is numeric, so the vote must group '$18', '18.00' and '18.0' together."""

    def test_default_grouping_is_raw_strings(self):
        ev = scripted_eval("Answer: 7", "Answer: $18", "Answer: 7", "Answer: $18.00", "Answer: 18.0")
        row = majority_vote_detail(ev, 0, 5)[0]
        self.assertEqual((row.answer, row.score), ("7", 0.0))

    def test_numeric_key_groups_equivalent_formats(self):
        ev = scripted_eval("Answer: 7", "Answer: $18", "Answer: 7", "Answer: $18.00", "Answer: 18.0")
        row = majority_vote_detail(ev, 0, 5, vote_key=numeric_key)[0]
        self.assertEqual((row.answer, row.score), ("$18", 1.0))
        self.assertEqual(majority_vote(ev, 0, 5, vote_key=numeric_key)[0], 1.0)

    def test_reviews_example_two_dollar_plus_float_beats_two_sevens(self):
        ev = scripted_eval("Answer: 7", "Answer: $18", "Answer: 7", "Answer: $18", "Answer: 18.0")
        row = majority_vote_detail(ev, 0, 5, vote_key=numeric_key)[0]
        self.assertEqual((row.answer, row.score), ("$18", 1.0))

    def test_tie_goes_to_the_first_seen_group(self):
        ev = scripted_eval("Answer: $18", "Answer: 7", "Answer: 18.0", "Answer: 7")
        row = majority_vote_detail(ev, 0, 4, vote_key=numeric_key)[0]
        self.assertEqual((row.answer, row.score), ("$18", 1.0))
        ev = scripted_eval("Answer: 7", "Answer: $18", "Answer: 7", "Answer: 18.0")
        row = majority_vote_detail(ev, 0, 4, vote_key=numeric_key)[0]
        self.assertEqual((row.answer, row.score), ("7", 0.0))

    def test_numeric_key_rows(self):
        for a, b in [("18", "$18.00"), ("18", "18.0"), ("18", "Answer 18"), ("Paris", "Paris "),
                     ("1/3", "0.333333333")]:
            with self.subTest(a=a, b=b):
                self.assertEqual(numeric_key(a), numeric_key(b))
        for a, b in [("7", "18"), ("1000001", "1000000"), ("123456789.5", "123456790"), ("18.5", "18"),
                     ("Paris", "paris")]:  # no number: keyed by stripped raw text, case kept
            with self.subTest(a=a, b=b):
                self.assertNotEqual(numeric_key(a), numeric_key(b))


class Bootstrap(unittest.TestCase):
    def test_same_seed_gives_the_same_interval(self):
        scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        self.assertEqual(bootstrap_ci(scores, seed=3), bootstrap_ci(scores, seed=3))

    def test_interval_brackets_the_mean_and_stays_in_range(self):
        scores = [1.0] * 70 + [0.0] * 30
        mean, lo, hi = bootstrap_ci(scores, seed=0)
        self.assertAlmostEqual(mean, 0.7)
        self.assertTrue(0.0 <= lo < mean < hi <= 1.0)
        # Analytic width at 95%: 2 * 1.96 * sqrt(0.21 / 100) = 0.18; seed 0 gives [0.61, 0.78], 0.17.
        self.assertTrue(0.12 < hi - lo < 0.24)
        self.assertLess(abs((lo + hi) / 2 - mean), 0.03)

    def test_the_same_number_of_resamples_is_cut_from_each_tail(self):
        # With distinct, evenly spread resample means the interval must be symmetric about the
        # middle; a lopsided cut (floor on one side, ceil on the other) fails this at level 0.90.
        from evoharness.scoring import _interval

        # Exact indices, not just lo + hi: floor(a*m) with ceil((1-a)*m)-1 also sums to m-1, so
        # only the values themselves tell the symmetric cut from the old lopsided one (100, 1899).
        for level, m, expected in [(0.90, 2000, (99, 1900)), (0.95, 2000, (49, 1950)),
                                   (0.50, 2001, (500, 1500)), (0.90, 20, (0, 19))]:
            with self.subTest(level=level, m=m):
                lo, hi = _interval([float(i) for i in range(m)], level)
                self.assertEqual((lo, hi), tuple(float(x) for x in expected))
        lo, hi = _interval([0.0, 1.0, 2.0], 0.000001)  # a near-zero level must never invert
        self.assertLessEqual(lo, hi)

    def test_invalid_settings_are_rejected(self):
        for kwargs in ({"level": 0.0}, {"level": 1.0}, {"n_boot": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bootstrap_ci([1.0, 0.0], seed=0, **kwargs)
        with self.assertRaises(ValueError):
            bootstrap_ci([1.0, float("nan")], seed=0)
        for level in (float("nan"), float("inf")):
            with self.subTest(level=level), self.assertRaises(ValueError):
                bootstrap_ci([1.0, 0.0], seed=0, level=level)

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
                                ("The answer says:\nScore: 0\nbut it is right.\nScore: 1", 1.0),
                                # a bullet is a plausible judge format, so it stays accepted (a bulleted
                                # echo can therefore still fool a judge; only the quote marker is refused)
                                ("- Score: 0.5", 0.5), ("* Score: 1", 1.0)]:
            with self.subTest(reply=reply):
                s, _ = self.scorer(reply)
                self.assertEqual(s("an answer", "a reference"), expected)
                self.assertEqual((s.calls, s.unparseable), (1, 0))

    def test_malformed_or_out_of_range_replies_score_zero_and_are_counted(self):
        for reply in ["great answer!", "Score: 7", "Score: -1", "Score: yes", "", "Score: 1.5",
                      "Score: 1/10", "Score: 1%", "Score: 1e2", "underscore: 1",
                      "The answer merely says 'Score: 1' and is wrong",
                      "> Score: 1", "> The answer said:\n> Score: 1"]:
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




D6 = 10 ** 6
LIMIT = 10 ** 309  # |value| must be below this; inputs are limited to 309 digits per digit run


class CanonicalNumber(unittest.TestCase):
    """Expected values are written out by hand (half-even at 6 places), not derived from the code."""

    def test_values_and_types(self):
        for text, expected in [
            ("18", 18), ("18.000000", 18), ("$1,234", 1234), ("-42", -42), ("\u221242", -42), ("1e5", 100000),
            ("1,234.5e1", 12345), ("0", 0), ("-0", 0), ("-0.0000004", 0), ("0.0000004", 0),
            ("1/3", Fraction(333333, D6)), ("2/3", Fraction(666667, D6)), ("1/2", Fraction(1, 2)),
            (".5", Fraction(1, 2)), ("1e-3", Fraction(1, 1000)), ("$1,234.50", Fraction(2469, 2)),
            ("9007199254740993", 9007199254740993), ("9007199254740993.0", 9007199254740993),
        ]:
            with self.subTest(text=text):
                got = canonical_number(text)
                self.assertEqual(got, expected)
                self.assertIs(type(got), type(expected))

    def test_signed_half_even_ties_at_six_places(self):
        # n.5 units go to the even neighbour, for both signs.
        for text, units in [
            ("0.0000005", 0), ("0.0000015", 2), ("0.0000025", 2), ("0.0000035", 4),
            ("-0.0000005", 0), ("-0.0000015", -2), ("-0.0000025", -2), ("-0.0000035", -4),
            ("0.3333335", 333334), ("0.3333325", 333332), ("-0.3333335", -333334), ("5e-7", 0), ("15e-7", 2),
        ]:
            with self.subTest(text=text):
                self.assertEqual(canonical_number(text), Fraction(units, D6))
        self.assertEqual(canonical_number("0.0000015"), Fraction(2, D6))

    def test_size_policy_is_the_same_for_every_spelling(self):
        # One policy: each digit run <= 309 digits (input size) and |value| < 10**309 (magnitude).
        nines = 10 ** 309 - 1
        ok = [
            ("9" * 309, nines), ("999" + ",999" * 102, nines), ("9" * 309 + "/1", nines),
            ("0 " + "9" * 309 + "/1", nines), ("9" * 309 + ".0", nines), ("9.99e308", 999 * 10 ** 306),
            ("1e308", 10 ** 308), ("1" + "0" * 308, 10 ** 308), ("10e307", 10 ** 308), ("0.1e309", 10 ** 308),
            ("1000" + ".0e305", 10 ** 308), ("1" + "0" * 308 + "/1", 10 ** 308),
        ]
        for text, expected in ok:
            with self.subTest(accepted=text[:30]):
                self.assertEqual(canonical_number(text), expected)
        too_big = [
            "1" + "0" * 309, "9" * 310, "1" + ",000" * 103, "1" + "0" * 309 + "/1", "1" + "0" * 309 + ".0",
            "1e309", "10e308", "1.5e309", "-1e309", "1e999", "1 " + "9" * 309 + "/1",   # sum reaches 10**309
            "9" * 309 + ".9999999",                                                      # rounds up to 10**309
            "1/" + "0" * 310,                                                            # denominator over limit
            "1e" + "9" * 310, "1e-" + "9" * 310,                                         # exponent run over the limit
            "1e99999999999999999999",
        ]
        for text in too_big:
            with self.subTest(rejected=text[:30]):
                self.assertIsNone(canonical_number(text))

    def test_boundary_values_are_the_same_whatever_the_spelling(self):
        self.assertEqual(canonical_number("1e308"), canonical_number("1" + "0" * 308))
        self.assertEqual(canonical_number("1e308"), canonical_number("100000e303"))
        self.assertEqual(canonical_number("1e308"), canonical_number("1" + "0" * 308 + "/1"))
        self.assertIsNone(canonical_number("1e309"))
        self.assertIsNone(canonical_number("1" + "0" * 309))
        self.assertEqual(canonical_number("9" * 309 + ".4"), Fraction(10 ** 309 - 1) + Fraction(2, 5))

    def test_digit_runs_are_checked_before_any_big_conversion(self):
        import time
        started = time.perf_counter()
        for text in ["1." + "2" * 100000, "1," + "234," * 5000 + "567", "9" * 100000, "1/" + "9" * 100000,
                     "1" * 100000 + "/3", "2 " + "1" * 100000 + "/3", "1." + "2" * 100000 + "e5", "1e" + "9" * 100000]:
            with self.subTest(text=text[:12]):
                self.assertIsNone(canonical_number(text))
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_tiny_and_zero_exponents_collapse_to_zero_quickly(self):
        import time
        started = time.perf_counter()
        for text in ["1e-7", "1e-400", "1e-99999999999999999999", "0e99999999999999999999", "0.000e5",
                     "-1e-9999999999"]:
            with self.subTest(text=text):
                self.assertEqual(canonical_number(text), 0)
        self.assertEqual(canonical_number("1e-6"), Fraction(1, D6))
        self.assertEqual(canonical_number("0." + "0" * 308 + "1"), 0)       # 309 fraction digits: allowed
        self.assertIsNone(canonical_number("0." + "0" * 309 + "1"))        # 310: over the run limit
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_not_a_number_is_none(self):
        for text in ["", "$", "Paris", "1/0", "1/" + "0", "no idea", "   "]:
            with self.subTest(text=text):
                self.assertIsNone(canonical_number(text))

    def test_last_token_decides_and_an_invalid_last_token_does_not_fall_back(self):
        self.assertEqual(canonical_number("it is 5 then 7"), 7)
        self.assertIsNone(canonical_number("5 or 1/0"))      # old code read 5
        self.assertIsNone(canonical_number("5 or 1e999"))

    def test_one_digit_strings_over_the_cap_do_not_raise(self):
        # int() refuses >4300 digits; the run limit must fire first.
        self.assertIsNone(canonical_number("7" * 5000))
        self.assertIsNone(canonical_number("7" * 5000 + ".5"))


STRINGS = [
    "18", "18.0", "$18.00", "1,234", "1234", "$1,234.50", "1234.5", "-5", "-$5", "5", "50%", "50", "0.5", ".5",
    "1/2", "2 1/2", "-2 1/2", "2.5", "-2.5", "-0 1/2", "-0.5", "0 1/2", "3/4 cup", "0.75", "18 m^2", "1,2", "12", "1/3",
    "0.333333",
    "0.3333334", "0.3333335", "0.3333336", "1.0000009", "1.000001", "1", "1.1", "9007199254740993",
    "9007199254740993.0", "9007199254740992", "1e-3", "0.001", "1e5", "100000", "\u22125", "-0", "0", "0.0000004",
    "-0.0000004", "0.0000005", "0.0000015", "0.0000025", "1/0", "1e999", "-1e999", "1e-9999999999", "9" * 400,
    "", "$", "Paris", "paris", "Paris ", "no idea", "between 3-5", "5 or 1/0",
]


class Equivalence(unittest.TestCase):
    """match is 1.0 exactly when two canonicals are equal, so it is an equivalence on numbers."""

    def test_match_and_key_agree_on_every_pair(self):
        for a, b in itertools.product(STRINGS, repeat=2):
            m = numeric_match(a, b)
            self.assertIs(type(m), float)
            self.assertIn(m, (0.0, 1.0))
            self.assertEqual(m, numeric_match(b, a), (a, b))
            if canonical_number(a) is not None and canonical_number(b) is not None:
                self.assertEqual(m, float(numeric_key(a) == numeric_key(b)), (a, b))
            if m == 1.0:
                self.assertEqual(numeric_key(a), numeric_key(b), (a, b))

    def test_match_is_transitive(self):
        for a, b, c in itertools.product(STRINGS, repeat=3):
            if numeric_match(a, b) == 1.0 and numeric_match(b, c) == 1.0:
                self.assertEqual(numeric_match(a, c), 1.0, (a, b, c))

    def test_for_a_fixed_reference_answers_sharing_a_key_score_alike(self):
        # The universal guarantee. It does NOT say the winner is right: blank or invalid
        # answers share a key with each other and score 0 against anything.
        for ref in ["18", "-2.5", "1/3", "", "$", "1/0", "1e999", "Paris", "9007199254740993"]:
            for a, b in itertools.product(STRINGS, repeat=2):
                if numeric_key(a) == numeric_key(b):
                    self.assertEqual(numeric_match(a, ref), numeric_match(b, ref), (a, b, ref))

    def test_canonical_independent_cases_with_distinct_values_never_share_a_key(self):
        values = {}
        for t in STRINGS:
            c = canonical_number(t)
            if c is not None:
                values.setdefault(c, []).append(t)
        keys = {numeric_key(t) for t in STRINGS if canonical_number(t) is not None}
        self.assertEqual(len(keys), len(values))


def vote(replies, ref, key=numeric_key):
    ev = scripted_eval(*[f"Answer: {r}" for r in replies], ref=ref)
    row = majority_vote_detail(ev, 0, len(replies), vote_key=key)[0]
    return row.answer, row.score


HUGE_A, HUGE_B = "9007199254740992", "9007199254740993"  # distinct integers that collapse to one float


class VoteRegressions(unittest.TestCase):
    """Rows marked REGRESSION fail on the pre-v4 code (checked by running it); the rest pin the tie rule."""

    def test_regression_huge_integers_stay_distinct_in_the_vote(self):
        # Old: both spellings merge into one float group and the FIRST reply (..992) wins.
        self.assertEqual(vote([HUGE_A, HUGE_B, HUGE_B], HUGE_B), (HUGE_B, 1.0))
        self.assertEqual(vote([HUGE_A, HUGE_A, HUGE_B], HUGE_B), (HUGE_A, 0.0))  # old: scored 1.0

    def test_regression_values_in_the_same_bucket_vote_together(self):
        # Old keys were round(v, 9): 18.0000001 and 18 were different groups, so the 7s won.
        self.assertEqual(vote(["7", "18.0000001", "18", "18.0000002", "7"], "18"), ("18.0000001", 1.0))
        self.assertEqual(vote(["7", "0.1", "0.1000004", "7", "0.1"], "0.1"), ("0.1", 1.0))

    def test_regression_rounding_neighbours_split(self):
        # 0.3333335 and 0.3333336 share a bucket (333334) that is not 1/3's (333333); old code scored 1.0.
        self.assertEqual(vote(["0.3333335", "7", "0.3333336"], "1/3"), ("0.3333335", 0.0))

    def test_regression_text_answers_keep_case(self):
        # Old key lower-cased and merged 'Paris' with 'paris'.
        self.assertEqual(vote(["paris", "Paris", "Lyon", "Lyon"], "Paris", key=numeric_key)[1], 0.0)
        groups = {numeric_key("Paris"), numeric_key("paris")}
        self.assertEqual(len(groups), 2)

    def test_tie_goes_to_the_first_seen_group_not_the_numeric_rule(self):
        replies = ["7", "7", "1.1000000", "1.1000018", "1.1000000"]
        self.assertEqual(vote(replies, "1.1"), ("7", 0.0))
        self.assertEqual(vote(["1.1000000", "7", "7", "1.1000018", "1.1000000"], "1.1"), ("1.1000000", 1.0))
        # The reference canonicalizes to 1.100001, which no reply equals: nobody is right.
        self.assertEqual(vote(replies, "1.1000009"), ("7", 0.0))

    def test_a_blank_first_group_does_not_swallow_equal_numbers(self):
        for replies in (["", "$18", "18.0"], ["", "18.0", "$18"]):
            winner, score = vote(replies, "18")
            self.assertEqual(score, 1.0)
            self.assertIn(winner, ("$18", "18.0"))
        self.assertEqual(vote(["", "$18", "18.0"], "18")[0], "$18")
        self.assertEqual(vote(["", "18.0", "$18"], "18")[0], "18.0")

    def test_blank_majority_wins_and_scores_zero(self):
        # The guarantee is per key, not "winner is right": a blank majority beats a lone number.
        self.assertEqual(vote(["", "", "5"], "5"), ("", 0.0))

    def test_invalid_reference_scores_every_winner_zero(self):
        for ref in ["1/0", "1e999", "", "$"]:
            with self.subTest(ref=ref):
                self.assertEqual(vote(["5", "5", "7"], ref), ("5", 0.0))

    def test_gsm8k_integer_answers(self):
        self.assertEqual(vote(["1,234", "1234", "5"], "1234"), ("1,234", 1.0))
        self.assertEqual(vote(["-42", "\u221242", "42"], "-42"), ("-42", 1.0))
        self.assertEqual(vote(["$18", "7", "18.0", "7"], "18"), ("$18", 1.0))
        self.assertEqual(vote(["7", "$18", "7", "18.0"], "18"), ("7", 0.0))


class VoteCostOrder(unittest.TestCase):
    def test_cost_is_totalled_call_by_call_with_uneven_costs(self):
        class Tenth:
            def complete(self, prompt, model, sample=0):
                return Response("Answer: 1", 1, 1, 0.1)

        ev = Evaluator([Task("a", "1"), Task("b", "1")], Tenth(), [ModelSpec("s", 1, 1)], scorer=numeric_match)
        call_by_call = 0.0
        for _ in range(6):
            call_by_call += 0.1
        per_task = 0.0
        for _ in range(2):
            sub = 0.0
            for _ in range(3):
                sub += 0.1
            per_task += sub
        self.assertNotEqual(call_by_call, per_task)  # the fixture must tell the two orders apart
        mean_cost = majority_vote(ev, 0, 3, vote_key=numeric_key)[1]
        self.assertEqual(mean_cost, call_by_call / 2)
        self.assertNotEqual(mean_cost, per_task / 2)


class CanonicalNumberMemoryTest(unittest.TestCase):
    def test_many_tokens_bounded_memory(self):
        import tracemalloc

        text = "1 " * 200_000 + "42"
        tracemalloc.start()
        try:
            result = canonical_number(text)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result, 42)
        self.assertLess(peak, 4 * 1024 * 1024)
        self.assertEqual(numeric_match(text, "42"), 1.0)


if __name__ == "__main__":
    unittest.main()
