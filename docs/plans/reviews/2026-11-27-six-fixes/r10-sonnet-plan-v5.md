Plan5 review (laptop evoharness HEAD 619bca4, venv .venv)
1. `found` at executor.py:117-120 is only in canonical_number. Other `found` (lines 62-64) is extract_answer's own local. Grep-before-edit claim holds.
2. Reproduced on current code: "1 "*200_000+"42" -> tracemalloc peak 48.03 MB (matches Codex). Result 42, numeric_match(t,"42")=1.0.
3. Proposed loop: peak 0.0022 MB (2 KB). text.replace is a no-op (no U+2212 in input), so no copy. With U+2212 present the copy is ~0.4-1.6 MB.
4. 8 MB gap: old 48 MB vs new ~0 MB, so margin both ways is large. Suggest 4 MB: still 12x under old, >1.5 MB headroom for a replace copy. Do not go below 2 MB. Note tracemalloc must be start()/stop() inside the test (try/finally) so it does not leak into other tests; assert the result too.
5. Test is not flaky by design (deterministic allocation counts), but the plan's own "fails on 5787ff2" probe is already satisfied by item 2.
6. F2: guard is at tests/test_scoring.py:543; 23 test methods follow it, 46 precede (69 total def test_). Direct run today reports 46; after move should be 69, equal to pytest --co. Plan's "record both numbers" is right. Module-level constants (D6, LIMIT) after the guard are fine to leave; only the guard moves.
7. Plan gap: F2 has no durable guard; acceptable, but a one-line verification command in plan is enough.
8. Minor: plan says "Peak O(len(text))"; actual is O(1) when no U+2212 present. Harmless.
VERDICT: GO WITH CHANGES (set threshold to 4 MB or lower, wrap tracemalloc in try/finally)
