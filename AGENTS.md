# Agent notes (Cursor + Claude Code)

## Evoharness swarms

For **second opinion / committee / debate / plan critique / tool audit**, use the
packaged skill and wrappers — do not invent an ad-hoc multi-agent chat.

- Skill (auto-discovered in this repo):
  - Cursor: `.cursor/skills/evoharness-swarm/SKILL.md`
  - Claude Code: `.claude/skills/evoharness-swarm/SKILL.md`
  - Canonical copy: `skills/evoharness-swarm/SKILL.md`
- Commands: `bin/swarm review|debate|audit|committee|ask …` (JSON by default)
- User-level install links as `eh-swarm*` so a personal `~/bin/swarm` is untouched

Routine coding stays with the primary coding agent. See the skill for triggers.

Optional machine-wide install: `scripts/install-agent-skills.sh`
