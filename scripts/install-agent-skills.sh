#!/usr/bin/env bash
# Install evoharness swarm skill + wrappers for Claude Code and/or Cursor.
# Project checkouts already ship .claude/skills and .cursor/skills; this copies
# into user skill dirs so other repos can trigger the same workflow.
#
# Wrappers install as eh-swarm* under ~/.local/bin so an existing ~/bin/swarm
# is never overwritten.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/skills/evoharness-swarm"
NAME="evoharness-swarm"
LINK_PREFIX="eh-swarm"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--claude] [--cursor] [--bin] [--all]

  --claude   Copy skill to ~/.claude/skills/${NAME}
  --cursor   Copy skill to ~/.cursor/skills/${NAME}
  --bin      Symlink wrappers into ~/.local/bin as ${LINK_PREFIX}*
  --all      All of the above (default if no flags)

Note: does not replace ~/bin/swarm if you already have one. Use ${LINK_PREFIX}
or the repo path bin/swarm from agent skills.
EOF
}

do_claude=0
do_cursor=0
do_bin=0
if [[ $# -eq 0 ]]; then
  do_claude=1
  do_cursor=1
  do_bin=1
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --claude) do_claude=1 ;;
      --cursor) do_cursor=1 ;;
      --bin) do_bin=1 ;;
      --all) do_claude=1; do_cursor=1; do_bin=1 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
  done
fi

if [[ ! -f "$SRC/SKILL.md" ]]; then
  echo "missing $SRC/SKILL.md" >&2
  exit 1
fi

install_skill() {
  local dest="$1"
  mkdir -p "$dest"
  cp "$SRC/SKILL.md" "$dest/SKILL.md"
  echo "installed skill -> $dest/SKILL.md"
}

if [[ $do_claude -eq 1 ]]; then
  install_skill "$HOME/.claude/skills/$NAME"
fi

if [[ $do_cursor -eq 1 ]]; then
  install_skill "$HOME/.cursor/skills/$NAME"
fi

if [[ $do_bin -eq 1 ]]; then
  mkdir -p "$HOME/.local/bin"
  ln -sfn "$ROOT/bin/swarm" "$HOME/.local/bin/$LINK_PREFIX"
  for suffix in review debate audit committee; do
    ln -sfn "$ROOT/bin/swarm-$suffix" "$HOME/.local/bin/${LINK_PREFIX}-$suffix"
  done
  echo "linked ~/.local/bin/${LINK_PREFIX}* -> $ROOT/bin/swarm*"
  if [[ -e "$HOME/bin/swarm" ]]; then
    echo "note: $HOME/bin/swarm left untouched (personal wrapper)."
  fi
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) echo "note: add \$HOME/.local/bin to PATH if ${LINK_PREFIX} is not found" ;;
  esac
fi
