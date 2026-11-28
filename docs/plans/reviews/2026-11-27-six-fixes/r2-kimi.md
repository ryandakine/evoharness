Source review only. **Tests UNVERIFIED:** the permitted pytest command was refused by the environment, so the claimed “86 tests passing” was not reproduced. Sandbox runtime behavior below is **UNVERIFIED**; no bwrap was executed.

### High

- **WP1 §§Cost/Arms — “matched cost” is not matched.** Four calls can cost less than three long calls; output tokens omit input tokens, and E/F mix different-sized models. Reading C/D/E against B3/B5 does not establish equal or lower compute. **Fix:** remove “matched-cost” claims; report input/output tokens, calls, latency, and model identity per arm. Compare accuracy at predeclared call/token budgets or describe results as an exploratory accuracy–cost table.

- **WP1 §§Data/Split — underpowered and leaky.** With three samples clustered within 50 test questions, an accuracy near 50% has a rough 95% half-width of `1.96√(.25/50)` ≈ **14 percentage points**—a **28-point-wide interval**. Treating 150 responses as independent would give a misleading ±8 points. A–E on 100 questions still have roughly ±10 points. Moreover, F is “searched on 30+20,” leaving validation unused for selection; A–E’s all-100 scores are not directly comparable with F’s 50. The stop rule inspects A on all 100, including F’s test set, then changes the benchmark. **Fix:** search on 30, select exactly one Pareto candidate on 20, freeze it, then evaluate every arm on the same 50. Run stop/model-switch decisions only on development data; preferably enlarge the held-out set and report inconclusive results honestly.

- **WP1 §Runner + `llm.py:77-80,291-292`; `fitness.py:82-96` — cache/sample collisions.** Repeated outer scoring of B3/B5 will reuse `majority_vote()` sample IDs `0..k-1`, producing identical cached votes rather than three repetitions. Cache keys also omit temperature, max tokens, generation options, and Ollama model digest; `APIBackend` ignores `sample` except for cache separation. **Fix:** derive unique composite sample IDs for every outer repetition, and namespace/invalidate caches using full generation configuration plus model digest. Add a regression test proving repeated votes make distinct calls.

- **WP3 — confinement boundary is incomplete.** A workspace equal to `$HOME` still exposes `.ssh`/`.env`; readable host `/etc` is exposed; MCP tools remain unsandboxed when enabled through CLI defaults. No PID/IPC isolation is specified. **Fix:** reject root/home workspaces, document that workspace contents are exposed, disable MCP for confined audit runs, minimally bind required `/etc` files, and add PID/IPC isolation and capability dropping.

- **WP5 — tools are not actually wired into execution.** `run_genome` requires `node_tools`; passing only `tool_registry` leaves per-node restrictions unused (`executor.py:298-313`). `GraphDistribution` has no template/tools field (`distribution.py:28-54`), and adding an argument to `random_genome` alone does not plumb tools through `ga.py:93-96`. `None` means *all registered tools*, not none. **Fix:** pass a map from `g.tools`, add deep-copied tools to distribution construction, plumb the tool table through `evolve`, and define/reject mixed-`None` crossover semantics explicitly.

### Medium

- **WP3 flags/compatibility — UNVERIFIED.** Minimal `/dev`, `/proc`, and explicit working-directory handling are absent. Git, pytest, external venvs, and user-local binaries may fail; read-only verifier workspaces conflict with pytest cache/bytecode writes. Network installs intentionally fail. **Fix:** add minimal `--dev`/`--proc`, `--chdir`, writable scratch space, and real git/pytest/venv integration fixtures. Refusing bash without working bwrap is correct, but probe bwrap by execution and resolve this lazily only when bash is requested.

- **WP2 — mixed preset fails as written.** `cli.py:264-269` requires global `--model` before preset loading. `bin/swarm` lacks `--models`, mixed aliases, and forwarding. The judge validation wording permits a judge equal to one solver when solvers differ. **Fix:** bypass/derive the default model for mixed, wire the wrapper, require exactly four entries, and reject `judge in solvers`. Define malformed/partial env/config behavior and fail before any request without exposing config contents.

- **WP5 — parallel mutation and budgets.** Summing in task order fixes arithmetic ordering, not shared tool/context races. Concurrent `CachedBackend` calls can collectively overshoot a spend cap. **Fix:** initially cut `task_workers`; otherwise force serial execution for mutating tools or isolate each task, and test concurrent budget exhaustion.

- **Ordering/code claims.** WP1 cannot complete arm E before WP2. `jfk_scores` already preserves tools through `g.copy()` (`swarm.py:174`). `split_tasks` defaults to 60/20/20, not 30/20/50. **Fix:** move WP2 before WP1 or defer E; pass explicit split fractions; narrow WP5’s claim.

- **Tests/rollback/scope.** Add tests for held-out freeze/non-overlap, outer-sample cache behavior, actual node-tool enforcement, aliasing/`None`, and bwrap-present-but-unusable. Use a fake HOME, never real SSH material. Each PR needs revert/feature-flag instructions; sandbox rollback must not silently restore insecure bash. Defer the LLM judge and parallel scoring as nonessential scope.

VERDICT: GO WITH CHANGES

---
delegate: provider=kimi model=k3.1-pro turns=28 mode=agent tokens in=231511 out=11809 session=7e7441de-5552-4445-a14f-ed37cfa9d44f write_dir=None
