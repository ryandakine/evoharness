# Changelog

## 0.2.0 (fork)

Focus: make the harness dependable against real model APIs, and usable from the
command line, without changing any of the optimization math. The offline demos
(`run_mock.py`, `run_methods.py`) produce byte-identical numbers before and after.

### Added
- `evoharness` command (`[project.scripts]`) with built-in swarms (`--preset solo |
  pipeline | debate | committee`), custom graphs from JSON (`--spec`), `--json` and
  `--quiet` output, `--parallel`, `--retries`, `--timeout`, `--env-file`, `--list-presets`,
  `--version`, and readable errors that name the failing agent.
- `examples/swarm_spec.json`: a documented custom-swarm spec.
- `Genome.to_dict()` / `Genome.from_dict()` for saving found graphs as JSON, and
  `Genome.validate_against(n_roles, n_models)`.
- `run_genome(..., max_workers=N)` and `Evaluator(max_workers=N)`: agents with no path
  between them run concurrently, one DAG layer at a time. Outputs, costs and call counts
  are identical to the sequential run.
- `CachedBackend(cache_path=...)`: persistent JSON-lines response cache, so repeated or
  crashed searches re-spend nothing. Malformed lines are skipped on load.
- `APIBackend(retries=2, retry_backoff=1.0)`: exponential backoff on rate limits,
  timeouts, connection errors, 5xx responses and empty replies (`EmptyResponse`).
  `is_transient(exc)` exposes the heuristic. Non-transient errors still fail fast.
- `AgentError` (with `.node` and `.model_name`) wraps a failed model call inside
  `run_genome`; `BudgetExceeded` passes through unwrapped.
- `MockBackend(model_names=[...])` for reproducible parallel mock runs.
- GitHub Actions CI: tests and ruff on Python 3.10-3.13. `dev` extra with pytest + ruff.
- Package metadata (readme, license, classifiers, URLs), `__version__`, `__all__`.

### Changed
- `extract_answer` is case-insensitive, tolerates markdown emphasis (`**Answer:** 42`,
  `**Answer**: 42`), accepts `Final Answer:`, falls back to the last LaTeX `\boxed{}`,
  and strips wrapper characters. The last-line fallback now skips trailing blank lines.
- Cache keys no longer include token prices. A cached response whose price changed is
  re-costed from its stored token counts instead of being fetched again.
  (Previously, correcting a price typo re-spent on every cached call.)
- `Genome` rejects non-integer indices and malformed edges with a clear `ValueError`.
- `crossover` on different-sized genomes raises `ValueError` instead of `AssertionError`;
  `evolve` validates `seeds` against `GAConfig` up front and no longer shares seed
  objects with its population.
- `CachedBackend` and `MockBackend` are thread-safe.
- `APIBackend` suppresses LiteLLM's "Give Feedback / Get Help" banner.
- `examples/run_swarm.py` is now a thin wrapper over the CLI and exits with its status.
- README rewritten: correct clone URL (the original pointed at a different repository),
  consistent naming, spec format, free-router caveat, library walkthrough.
- `run_mock.py` prints the best graph as JSON.

### Fixed
- The `debate` demo crashed with a traceback on any empty reply from a shared/free router;
  it now retries, and if it still fails it says which agent failed and why.
- The example's lint issues (star imports, unsorted imports, lambda assignments).

### Unchanged on purpose
- NSGA-II, PSO/G-Decode, REINFORCE, DAG-Shapley, JFK and prompt-reflection logic.
- The `deterministic=True` contract on `Evaluator.attribute`.
- Nominal-cost accounting: cache hits are free for the budget but still report cost.
