### 1. Benchmark Design (WP1)

- **HIGH · WP1 (`CachedBackend.key` silently collapses Arms B/C/D/E):** `CachedBackend.key` (`llm.py:88-90`) hashes `model.name|api_base|sample|prompt`—no node index, no generation params.
  1. In Arm C (same-model committee), parallel solvers with the same role/model send identical prompts with the same `sample`, so solvers 1–2 hit solver 0's cache and return identical text—turning the committee into 3 clones of 1 call.
  2. `majority_vote` (`fitness.py:86-98`) hardcodes `sample=0..k-1`, so 3 repeated runs of B3/B5 return identical cached answers with zero variance, and overlap Arm A's `sample=0..2`.
  - **Fix:** Pass disjoint sample offsets (`arm_offset + repeat * k + s` or per-node `sample = repeat * n_nodes + node_idx`), or include `node_id` and generation config (temperature, `max_tokens`, model digest) in the cache key.
- **HIGH · WP1 (Underpowered sample size & mismatched split):** At $p \approx 0.80$, $N=100$ gives 95% CI $\approx \pm 8$ pp (~16 pp width); Arm F's $N=50$ test split gives $\approx \pm 11$ pp (~22 pp width). Three repeats shrink sampling noise, not task variance. Reporting A–E on all 100 and F on 50 makes cross-arm table comparisons invalid.
  - **Fix:** Report all arms (and paired diffs) on the same 50-question held-out split (or use $N \ge 300$ if comparing 3–5 pp differences), and apply the 95% stop rule on train/val only.
- **MEDIUM · WP1 (Unfair "matched cost"):** Counting calls + output tokens ignores 3–4× larger judge input tokens, unequal model sizes in Arm E (3B vs 9B), and Arm F's ~1,600-eval search cost.
  - **Fix:** Report input tokens, output tokens, wall-clock latency, and search calls separately.

### 2. Wrong Code Claims & Broken Steps

- **HIGH · WP1 (`--budget-usd 0` bricks local runs):** `CachedBackend.complete` (`llm.py:124-125`) raises `BudgetExceeded` when `spent >= budget`. With `budget_usd=0`, `0.0 >= 0.0` blocks the very first call—even for $0 local models.
  - **Fix:** Validate `price_in == price_out == 0` in `run.py` when `--budget-usd 0`, and pass `budget_usd=None` (or allow $0-priced models in `CachedBackend` when `budget == 0`).
- **HIGH · WP1/WP4 (Missing plumbing):** `RunResult` (`executor.py:30-37`) and `majority_vote` don't record token counts, and neither `evaluate_frozen` (`evaluation.py:55-68`) nor `majority_vote` returns per-task scores needed by `paired_bootstrap`. Also WP1 references "both models as genes" in Arm F despite naming only one base model.
  - **Fix:** Add `input_tokens`/`output_tokens` to `RunResult`, return per-task scores from `score_samples`/`majority_vote`, and name Arm F's model list explicitly.
- **HIGH · WP5 (`g.tools` is dead in `run_genome`):** `run_genome` (`executor.py:225,241`) only reads the `node_tools` kwarg and ignores `g.tools`. Passing `tool_registry`/`context`/`max_tool_turns` from `Evaluator` won't enforce per-node tools.
  - **Fix:** Have `run_genome` default `node_tools` from `g.tools` when `node_tools is None` (and include `allowed_tools` in `node_cache` keys).
- **MEDIUM · Order & Dependencies:** WP1 includes Arm E (from WP2) yet schedules `WP1 → WP2`.
  - **Fix:** Reorder to `WP6 → WP4 → WP2 → WP1 → WP3 → WP5`.

### 3. Bash Sandbox (WP3)

- **HIGH · WP3 (Missing `bwrap` flags & `shell=True`):**
  1. `builtin.py:63-65` uses `shell=True`; wrapping with `bwrap` MUST switch to `shell=False` with `["bwrap", ..., "--", "/bin/bash", "-c", command]`.
  2. Omitting `--proc /proc`, `--dev /dev`, `--unshare-pid`, `--clearenv`, and `HOME=/tmp` breaks Python/git/pytest and weakens process isolation and timeout cleanup.
  3. Checking `bwrap` on `PATH` isn't enough (AppArmor/containers block userns); probe with `bwrap --ro-bind / / true`.
- **HIGH · WP3 (Per-node RO/RW & toolchain breakage):** Resolving sandbox "once at start-up" contradicts per-node RO/RW workspace in `audit`; pass `read_only` via `ExecutionContext`. Unmounting `$HOME` breaks `mise`, `~/.local`, and `$HOME`-based Python venvs (`pytest`/`python`).
  - **Fix:** Pass per-node `read_only` in `ExecutionContext`, bind `sys.prefix`/`sys.base_prefix` (or `--ro-bind` PATH toolchain dirs) read-only, and mount minimal `/etc` files (or document `/etc` RO exposure).
- **LOW · WP3 (Fail-closed default):** Refusing bash without `bwrap` unless `--sandbox none` is right for untrusted repos, as long as file tools still run.

### 4. Optimizers & Parallelism (WP5)

- **HIGH · WP5 (Non-deterministic `MockBackend` + budget race):** `MockBackend.complete` (`llm.py:276-280`) assigns model indices in first-seen order when `model_names` is omitted, so `task_workers > 1` changes scores on multi-model genomes. `CachedBackend` releases `_lock` before calling `inner`, letting `N` threads overshoot `budget_usd`, and `complete_messages` doesn't cache at all (`llm.py:134-156`).
  - **Fix:** Pass `model_names` explicitly (or hash `model.name` stably in tests), handle `None` vs `list` `tools` in `crossover`, wire tools into `Evaluator.attribute()`, and cap `task_workers * max_workers`.

### 5. Mixed-Model Preset (WP2)

- **MEDIUM · WP2 (Dead check, `--models` metadata gap, secret hygiene):**
  1. "judge equal to every solver" is unreachable once ≥2 distinct solvers exist (and conflicts with Arm E's self-judging `qwen3:8b`). Pick and enforce one rule.
  2. `--models a,b,c,judge` lacks `api_base`/`api_key_env`/prices for mixed providers.
  3. Reject raw `api_key` fields in `models.json` and fail fast on missing `api_key_env` before calling solver 0.

### 6. Scope, Tests & Rollback

- **MEDIUM · Scope:** Cut `llm_judge_scorer` from WP4 (unused) and defer Arm F until A–E results justify overnight search.
- **MEDIUM · Missing tests & cache rollback:** Add tests for `budget_usd=0` on $0 vs priced models, same-model committee cache isolation, `g.tools` enforcement, `bwrap` probe failure/`shell=False`, and fail-fast missing credentials. Version `cache.jsonl` or key it by generation params + model digest to prevent stale-cache contamination.

**UNVERIFIED** (files not provided): `cli.py` presets/flags, `swarm.py` (`g_decode`, `jfk_scores`), `distribution.py` (`GraphDistribution`), `tools/base.py` (`ExecutionContext`, `ToolRegistry` thread-safety), `tests/`, `pyproject.toml`, `README.md`, `bin/swarm`, `examples/run_methods.py`.

**VERDICT: GO WITH CHANGES**
