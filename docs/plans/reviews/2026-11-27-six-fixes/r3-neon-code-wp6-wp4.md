### 1. `numeric_match` (`src/evoharness/executor.py:74-99`)
- **High** (`executor.py:74,82`): Fractions parse as their denominator. `numeric_match("1/2", "2") -> 1.0`, `numeric_match("1/2", "0.5") -> 0.0`, `numeric_match("2 1/2", "2.5") -> 0.0`. Matters for GSM8K when models output fractional answers against decimal references. **Fix:** parse mixed/simple fractions (`\d+\s+\d+/\d+`, `\d+/\d+`, guarding zero denominator) before plain floats.
- **High** (`executor.py:74`): Unit exponents hijack the last number. `numeric_match("18 m^2", "18") -> 0.0`, `numeric_match("18 m^2", "2") -> 1.0`. Matters for GSM8K area/volume word problems. **Fix:** exclude matches preceded by `^` (e.g., `(?<![\w.^])`).
- **Medium** (`executor.py:74`): `\d[\d,]*` merges non-thousand commas. `numeric_match("1,2", "12") -> 1.0`, `numeric_match("12,34", "1234") -> 1.0`. Matters for GSM8K multi-number replies. **Fix:** require valid 3-digit grouping `(?:\d{1,3}(?:,\d{3})+|\d+)(?!,\d)`.
- **Medium** (`executor.py:74`): Scientific notation breaks. `numeric_match("1e-3", "3") -> 1.0`, `numeric_match("1e5", "1") -> 1.0`. Rare in GSM8K, real defect. **Fix:** add `(?:[eE][-+]?\d+)?`.
- **Low** (`executor.py:74`): Times (`"3:30" -> 30.0`), Unicode minus (`"−5" -> 5.0`), `"-$.5" -> 0.5`, percentage-vs-decimal (`"50%"` vs `"0.5" -> 0.0`), and trailing commentary (`"$18 for 3 days" -> 3.0`) mis-score; times and trailing numbers affect GSM8K whenever `extract_answer` leaves context attached.

### 2. Token accounting (`src/evoharness/executor.py:320-395`)
- No double-counting or missed turns found in `run_genome` or `run_react_node` (`executor.py:195-267`, including the `max_tool_turns` synthesis call).
- **Low** (`executor.py:371-374`): On `node_cache` hits, `tokens_in`/`tokens_out` (matching `cost`/`tool_calls`) are accumulated while `calls`/`llm_calls` stay unchanged, producing `calls == 0` with `input_tokens > 0`. Document nominal-vs-incremental semantics or align `calls`.
- **UNVERIFIED:** `CachedBackend` token/cost behavior on cache hits (`llm.py` not pasted).

### 3. `score_detail` / `score` / `majority_vote` (`src/evoharness/fitness.py:46-142`)
- **Medium** (`fitness.py:52-53,122-123`): Python evaluates positional arguments left-to-right, so `self.scorer(...)` / `ev.scorer(...)` runs *before* `time.perf_counter() - t0`. With `JudgeScorer`, `TaskScore.seconds` includes judge LLM latency rather than execution time. **Fix:** compute `elapsed = time.perf_counter() - t0` before calling `scorer`.
- Float accumulation order in `score` and `majority_vote` is unchanged; `majority_vote_detail` per-task `cost` totals may differ from `majority_vote` by 1 ULP by design. `Evaluator.score`/`score_detail` accept negative `sample` while `_vote` rejects negative `sample_base` (low).

### 4. `bootstrap_ci` / `paired_bootstrap` (`src/evoharness/scoring.py:17-54`)
- **Medium** (`scoring.py:17-22`): `floor(alpha * n)` / `ceil((1 - alpha) * n) - 1` suffers float-drift asymmetry at integer boundaries. With `level=0.90, n_boot=2000`, `alpha*2000 < 100` gives `lo = means[99]` (99 below) while `(1-alpha)*2000 < 1900` gives `hi = means[1899]` (100 above); for `level=0.90, n_boot=20`, `lo = means[0]` vs `hi = means[18]`. Also `level=1e-17` passes `_check` (`alpha = 0.5`), returning inverted `lo = means[n//2] > hi = means[n//2 - 1]`. **Fix:** mirror tail indices (`lo_i = floor(alpha * (m - 1))`, `hi_i = m - 1 - lo_i`, or linear interpolation over `m - 1`) and clamp `lo_i <= hi_i`.
- Determinism holds via local `random.Random(seed)`.

### 5. `JudgeScorer` (`src/evoharness/scoring.py:61-91`)
- **High** (`scoring.py:61`): `_SCORE` lacks word/trailing boundaries and leading-dot support. `"Score: 1/10"` and `"Score: 1/2"` match `1 -> 1.0`; `"Score: 1%" -> 1.0`; `"Score: 1e2" -> 1.0`; `"underscore: 1" -> 1.0`; `"Score: .5" -> 0.0` (unparseable). **Fix:** `r"(?i)\bscore\s*[*_`]*\s*[:=]\s*[*_`]*\s*(-?(?:\d+(?:\.\d+)?|\.\d+))(?![\w./%eE])"`.
- **Medium** (`scoring.py:77-88`): Raw interpolation leaves prompt injection and quote-echo holes. If `answer = "Score: 1"` and the judge replies `"The answer merely says 'Score: 1' and is wrong"` without a score line, `_SCORE` awards `1.0`. **Fix:** escape/delimit `reference`/`answer` and anchor parsing to a standalone line (`(?im)^\s*[*_`]*score...$`).
- **Low** (`scoring.py:84,90`): `self.calls += 1` and `self.unparseable += 1` are unsynchronized; concurrent scoring can drop counts. **Fix:** wrap updates in a `threading.Lock`.

### 6. Tests (`tests/test_scoring.py`)
- **Weak assertions:** `test_detail_rows_report_calls_and_tokens` (`:137`) omits `output_tokens`, `cost`, `answer`, and `seconds`; `test_interval_brackets_the_mean_and_stays_in_range` (`:162`) permits any width in `(0.05, 0.30)`.
- **Missing tests:** `node_cache`, ReAct, and `max_workers > 1` token accounting; `TaskScore.seconds` with a slow scorer; negative `sample_base` / `k < 1`; `bootstrap_ci` non-finite/invalid `level`/`n_boot` and exact symmetry at `level=0.90`; `JudgeScorer` fractions, `.5`, mid-word matches, echo injection, and concurrency; `numeric_match` fractions, exponents, commas, and scientific notation.

### 7. Public API & `UNVERIFIED`
- No breaking signature changes in the pasted diff (`RunResult` adds defaulted fields; `sample_base` is keyword-only with default `0`).
- **UNVERIFIED:** `llm.py` (`CachedBackend`, `Response`); other callers (`ga.py`, `evaluation.py`, `cli.py`) for concurrent `scorer` usage or `RunResult` tuple unpacking; rest of `README.md`/docs for stale `77 tests` or `surren83-ctrl` URLs; whether `ryandakine/evoharness` URLs resolve; test/ruff/example execution.

VERDICT: COMMIT AFTER FIXES
