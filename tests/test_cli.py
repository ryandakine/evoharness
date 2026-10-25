"""The evoharness command, with the LiteLLM transport mocked."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evoharness import ModelSpec
from evoharness.cli import PRESETS, load_spec, main

EXAMPLE_SPEC = Path(__file__).resolve().parents[1] / "examples" / "swarm_spec.json"


def reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


class SpecLoading(unittest.TestCase):
    default = ModelSpec("provider/default", 1, 2)

    def test_presets_are_valid_specs(self):
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                genome, roles, models = load_spec(preset, self.default)
                self.assertEqual(genome.active_nodes(), list(range(genome.n)), "preset has unreachable agents")
                self.assertEqual(models, [self.default])
                self.assertIn("Answer:", roles[genome.roles[-1]])

    def test_example_spec_file_loads(self):
        with EXAMPLE_SPEC.open(encoding="utf-8") as fh:
            genome, roles, models = load_spec(json.load(fh), self.default)
        self.assertEqual(genome.n, 4)
        self.assertEqual(len(roles), 4)

    def test_models_table_and_node_objects(self):
        spec = {"roles": ["a", "b"], "nodes": [{"role": 0, "model": 1}, {"role": 1}],
                "edges": [[0, 1]],
                "models": ["provider/x", {"name": "provider/y", "price_in": 5, "api_base": "http://h/v1"}]}
        genome, roles, models = load_spec(spec, self.default)
        self.assertEqual(genome.models, [1, 0])
        self.assertEqual(models[0], ModelSpec("provider/x", 1, 2))
        self.assertEqual(models[1], ModelSpec("provider/y", 5, 2, "http://h/v1", None))

    def test_bad_specs_raise_readable_errors(self):
        bad = [
            ({}, "roles"),
            ({"roles": ["a"]}, "nodes"),
            ({"roles": ["a"], "nodes": ["x"]}, "node 0"),
            ({"roles": ["a"], "nodes": [{"role": "0"}]}, "integer"),
            ({"roles": ["a"], "nodes": [0, 0], "edges": [[1, 0]]}, "edges"),
            ({"roles": ["a"], "nodes": [1]}, "role index"),
            ({"roles": ["a"], "nodes": [{"role": 0, "model": 1}]}, "model index"),
            ({"roles": ["a"], "nodes": [0], "models": [1]}, "model"),
        ]
        for spec, message in bad:
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, message):
                load_spec(spec, self.default)


class Command(unittest.TestCase):
    def setUp(self):
        def echo(**kw):
            return reply(f"reply to: {kw['messages'][0]['content'][:30]}\nAnswer: 42")
        self.completion = Mock(side_effect=echo)
        patcher = patch.dict(sys.modules, {"litellm": SimpleNamespace(completion=self.completion),
                                           "dotenv": SimpleNamespace(load_dotenv=Mock())})
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {"LLM_MODEL": "", "LLM_API_BASE": "", "LLM_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = main(list(argv))
            except SystemExit as exc:  # argparse errors
                code = exc.code
        return code, out.getvalue(), err.getvalue()

    def test_default_preset_text_output(self):
        code, out, err = self.run_cli("--model", "provider/m", "--prompt", "what is 6*7")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.completion.call_count, 4)
        self.assertIn("--- Agent 0 [provider/m]", out)
        self.assertIn("FINAL: 42", out)
        self.assertIn("Cost tracking disabled", out)
        self.assertIn("preset 'debate'", err)

    def test_json_and_quiet_modes(self):
        code, out, _ = self.run_cli("--model", "provider/m", "--preset", "pipeline", "--prompt", "p", "--json",
                                    "--price-in", "1", "--price-out", "2")
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertEqual(report["answer"], "42")
        self.assertEqual(report["calls"], 3)
        self.assertEqual(report["preset"], "pipeline")
        self.assertEqual(report["genome"]["edges"], [[0, 1], [1, 2]])
        self.assertAlmostEqual(report["cost"], 3 * (10 * 1 + 5 * 2) / 1e6)
        self.assertEqual(set(report["outputs"]), {"0", "1", "2"})
        code, out, _ = self.run_cli("--model", "provider/m", "--preset", "solo", "--prompt", "p", "--quiet")
        self.assertEqual((code, out.strip()), (0, "42"))

    def test_spec_file_and_per_node_models(self):
        spec = {"roles": ["solve", "finish with Answer:"], "nodes": [{"role": 0, "model": 1}, {"role": 1}],
                "edges": [[0, 1]], "models": ["provider/a", "provider/b"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swarm.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            code, out, _ = self.run_cli("--model", "ignored/m", "--spec", str(path), "--prompt", "p", "--json")
        self.assertEqual(code, 0)
        self.assertEqual([c.kwargs["model"] for c in self.completion.call_args_list], ["provider/b", "provider/a"])
        self.assertIsNone(json.loads(out)["preset"])

    def test_usage_errors(self):
        self.assertEqual(self.run_cli("--prompt", "p")[0], 2)  # no model anywhere
        self.assertEqual(self.run_cli("--model", "m")[0], 2)  # no prompt
        self.assertEqual(self.run_cli("--model", "m", "--prompt", "p", "--spec", "/nonexistent.json")[0], 2)
        self.assertEqual(self.run_cli("--model", "m", "--prompt", "p", "--parallel", "0")[0], 2)
        self.completion.assert_not_called()

    def test_agent_failure_is_reported_with_hint(self):
        self.completion.side_effect = None
        self.completion.return_value = reply("")
        code, _, err = self.run_cli("--model", "provider/m", "--preset", "solo", "--prompt", "p", "--retries", "0")
        self.assertEqual(code, 1)
        self.assertIn("agent 0 (provider/m) failed", err)
        self.assertIn("hint:", err)

    def test_list_presets_and_env_defaults(self):
        code, out, _ = self.run_cli("--list-presets")
        self.assertEqual(code, 0)
        for name in PRESETS:
            self.assertIn(name, out)
        with patch.dict(os.environ, {"LLM_MODEL": "env/model", "LLM_API_BASE": "http://env/v1"}):
            code, _, _ = self.run_cli("--prompt", "p", "--preset", "solo")
        self.assertEqual(code, 0)
        self.assertEqual(self.completion.call_args.kwargs["model"], "env/model")
        self.assertEqual(self.completion.call_args.kwargs["api_base"], "http://env/v1")


if __name__ == "__main__":
    unittest.main()
