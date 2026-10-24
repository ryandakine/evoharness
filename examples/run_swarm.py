"""Run a four-agent swarm with any LiteLLM-supported text model API."""
import argparse
import os
import sys
from pathlib import Path

from evoharness import APIBackend, Genome, ModelSpec, Task, run_genome


def optional_int(value):
    return None if value == 'omit' else int(value)


def optional_float(value):
    return None if value == 'omit' else float(value)


def main():
    config_path = Path(__file__).resolve().parents[1] / '.env'
    if config_path.is_file():
        try:
            from dotenv import load_dotenv
        except ImportError:
            raise SystemExit('Install API dependencies: python -m pip install -e ".[api]"')
        # Keep environment overrides; do not expand ${...} inside secrets.
        load_dotenv(config_path, override=False, interpolate=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=os.environ.get('LLM_MODEL'), help='LiteLLM provider/model; or LLM_MODEL')
    parser.add_argument('--api-base', default=os.environ.get('LLM_API_BASE') or None, help='Custom base URL; or LLM_API_BASE')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--price-in', type=float, default=0, help='USD per million input tokens')
    parser.add_argument('--price-out', type=float, default=0, help='USD per million output tokens')
    parser.add_argument('--max-tokens', type=optional_int, default=1024, help='Output limit, or omit')
    parser.add_argument('--temperature', type=optional_float, default=0.7, help='Temperature, or omit')
    args = parser.parse_args()
    if not args.model:
        parser.error('set LLM_MODEL or pass --model provider/model')
    if args.price_in < 0 or args.price_out < 0:
        parser.error('token prices must be nonnegative')
    roles = [
        'Solve the task carefully and explain your solution.',
        'Develop an independent alternative solution.',
        'Review the supplied solutions, correct errors, and resolve disagreements.',
        "Produce the final response. Finish with 'Answer: <final answer>'.",
    ]
    swarm = Genome([0, 1, 2, 3], [0, 0, 0, 0], {(0, 2), (1, 2), (2, 3)})
    backend = APIBackend(api_base=args.api_base, api_key=os.environ.get('LLM_API_KEY') or None,
                         max_tokens=args.max_tokens, temperature=args.temperature)
    result = run_genome(swarm, Task(args.prompt, ''), backend,
                        [ModelSpec(args.model, args.price_in, args.price_out)], roles)
    for node, output in result.outputs.items():
        print(f'\n--- Agent {node} ---\n{output}')
    print(f'\nFINAL: {result.answer}\nModel calls: {result.calls}')
    if args.price_in or args.price_out:
        print(f'Nominal inference cost: ${result.cost:.6f}')
    else:
        print('Cost tracking disabled: supply token prices with --price-in and --price-out.')


if __name__ == '__main__':
    # Windows terminals may default to a code page that cannot render model text.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    main()
