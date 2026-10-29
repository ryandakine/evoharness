---
name: evoharness-swarm
description: >
  Run reproducible multi-agent LLM swarms via the evoharness CLI wrappers for
  second opinions, committee judgment, adversarial debate, plan critique, and
  tool-using code audit. Use when the user asks for a second opinion, committee
  review, debate this, critique this plan, adversarial review, swarm review, or
  multi-agent judgment — not for routine coding, edits, or Taskmaster work.
  Prefer bin/swarm (or eh-swarm) over inventing ad-hoc chat subagents for those jobs.
---

# Evoharness swarm

Use the packaged wrappers. Do **not** reimplement a swarm in chat when this applies.

## When to use

- Second opinion / committee judgment on a plan, design, or claim
- Adversarial debate between independent solvers
- Tool-using audit of a repo path (`audit` preset)
- Closed-form hard questions where multiple solvers + a judge help

## When not to use

- Ordinary implement / fix / refactor / debug loops (use the coding agent)
- Simple factual lookups
- Tasks that need interactive IDE judgment across many edit turns

## Commands

Resolve the binary in this order:

1. `<evoharness-repo>/bin/swarm` when this repo (or a checkout) is available
2. `eh-swarm` on `PATH` (from `scripts/install-agent-skills.sh`)
3. Do **not** assume a personal `~/bin/swarm` API; prefer the repo wrapper

```bash
# Default second opinion — committee (JSON on by default)
bin/swarm review "Your question or pasted plan"
bin/swarm -c "…"                  # same preset, short flag

# Debate / pipeline / single agent
bin/swarm debate --prompt-file path/to/plan.md
bin/swarm -d -f plan.md "Critique the attached plan"
bin/swarm pipeline "..."
bin/swarm ask "..."

# Tool-using audit (builtin bash + files); confine with --cwd
bin/swarm audit --cwd "$PWD" "Audit tests and report failures"
bin/swarm -a --cwd "$PWD" "…"

# Human-readable final answer only
bin/swarm review -q "..."
```

Stdin works: `git diff main...HEAD | bin/swarm committee`

## Agent procedure

1. Choose preset (`review`/`committee`, `debate`, `audit`, …).
2. Run the wrapper above. JSON is the default (omit `-q` / `--text`).
3. Read final / answer fields from the JSON; summarize for the user.
4. Do not claim the swarm “fixed” code unless `audit` ran with tools and you verified the result.

## Requirements

- `python -m pip install -e ".[api]"` in the evoharness checkout (uses `.venv` when present).
- API key via `.env` (`LLM_API_KEY`, `LLM_MODEL`) or the process environment — never embed keys in commands.
- Pin a real model; avoid free auto-routers for anything you need to reason about.

## Specs

Custom graphs: `examples/swarm_spec.json` and `bin/swarm run --spec … --prompt … --json`.
