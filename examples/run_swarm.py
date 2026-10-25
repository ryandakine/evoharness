"""Run a swarm with any LiteLLM-supported text model API.

Thin wrapper around the ``evoharness`` command that loads the project-root .env.
After ``pip install -e ".[api]"`` you can call ``evoharness --prompt "..."`` directly.
"""
import sys
from pathlib import Path

from evoharness.cli import main

if __name__ == '__main__':
    # Windows terminals may default to a code page that cannot render model text.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    sys.exit(main(env_file=Path(__file__).resolve().parents[1] / '.env'))
