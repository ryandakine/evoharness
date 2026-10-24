"""Exercise the provider adapter without installing an SDK or sending requests."""
import importlib
import os
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from evoharness import APIBackend, CachedBackend, Genome, ModelSpec, Task, run_genome


def reply(content='Answer: yes', usage=True):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20) if usage else None)


class APITests(unittest.TestCase):
    def setUp(self):
        self.completion = Mock(return_value=reply())
        self.load_dotenv = Mock()
        self.modules = patch.dict(sys.modules, {
            'litellm': SimpleNamespace(completion=self.completion),
            'dotenv': SimpleNamespace(load_dotenv=self.load_dotenv),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_native_provider_and_usage_pricing(self):
        backend = APIBackend()
        result = backend.complete('question', ModelSpec('provider/model', 2, 10), sample=8)
        params = self.completion.call_args.kwargs
        self.assertEqual(params['model'], 'provider/model')
        self.assertEqual(params['messages'], [{'role': 'user', 'content': 'question'}])
        self.assertFalse(params['stream'])
        self.assertEqual(params['num_retries'], 0)
        self.assertNotIn('api_key', params)
        self.assertNotIn('seed', params)
        self.assertEqual(result.text, 'Answer: yes')
        self.assertAlmostEqual(result.cost, 0.0004)

    def test_custom_endpoint_and_model_credentials(self):
        with patch.dict(os.environ, {'TEST_PROVIDER_KEY': 'test-secret'}):
            backend = APIBackend(api_base='http://default/v1', api_key='default-secret')
            backend.complete('q', ModelSpec('openai/custom', 0, 0, 'http://other/v1', 'TEST_PROVIDER_KEY'))
        params = self.completion.call_args.kwargs
        self.assertEqual(params['api_base'], 'http://other/v1')
        self.assertEqual(params['api_key'], 'test-secret')

    def test_endpoint_override_does_not_forward_default_secret(self):
        backend = APIBackend(api_base='http://default/v1', api_key='default-secret')
        backend.complete('q', ModelSpec('openai/custom', 0, 0, 'http://other/v1'))
        self.assertNotIn('api_key', self.completion.call_args.kwargs)

    def test_omit_unsupported_generation_options(self):
        backend = APIBackend(max_tokens=None, temperature=None, extra_params={'max_completion_tokens': 256})
        backend.complete('q', ModelSpec('provider/reasoner', 0, 0))
        params = self.completion.call_args.kwargs
        self.assertNotIn('max_tokens', params)
        self.assertNotIn('temperature', params)
        self.assertEqual(params['max_completion_tokens'], 256)
        with self.assertRaises(ValueError):
            APIBackend(extra_params={'model': 'wrong'})

    def test_content_blocks_and_invalid_responses(self):
        backend = APIBackend()
        model = ModelSpec('provider/model', 0, 0)
        self.completion.return_value = reply([{'type': 'text', 'text': 'Answer: '}, {'type': 'text', 'text': 'yes'}])
        self.assertEqual(backend.complete('q', model).text, 'Answer: yes')
        for response in (reply(None), reply(''), reply(usage=False), SimpleNamespace(choices=[])):
            self.completion.return_value = response
            with self.assertRaises(ValueError):
                backend.complete('q', model)

    def test_missing_named_key_fails_before_request(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'TEST_MISSING_KEY'):
                APIBackend().complete('q', ModelSpec('provider/model', 0, 0, api_key_env='TEST_MISSING_KEY'))
        self.completion.assert_not_called()

    def test_cache_distinguishes_same_model_different_routes_and_prices(self):
        backend = CachedBackend(APIBackend())
        for endpoint in ('http://one/v1', 'http://two/v1'):
            backend.complete('q', ModelSpec('openai/model', 1, 1, endpoint))
        backend.complete('q', ModelSpec('openai/model', 1, 1, 'http://two/v1'))
        backend.complete('q', ModelSpec('openai/model', 2, 2, 'http://two/v1'))
        self.assertEqual(self.completion.call_count, 3)
        self.assertEqual(backend.hits, 1)

    def test_mixed_provider_swarm(self):
        models = [ModelSpec('provider-a/model', 0, 0), ModelSpec('provider-b/model', 0, 0)]
        g = Genome([0, 0], [0, 1], {(0, 1)})
        result = run_genome(g, Task('q', 'yes'), APIBackend(), models, ['solve'])
        self.assertEqual(result.calls, 2)
        self.assertEqual([c.kwargs['model'] for c in self.completion.call_args_list],
                         ['provider-a/model', 'provider-b/model'])

    def test_provider_errors_propagate_without_retry(self):
        self.completion.side_effect = RuntimeError('provider unavailable')
        with self.assertRaisesRegex(RuntimeError, 'provider unavailable'):
            APIBackend().complete('q', ModelSpec('provider/model', 0, 0))
        self.assertEqual(self.completion.call_count, 1)

    def test_missing_optional_dependency_is_actionable(self):
        error = ModuleNotFoundError("No module named 'litellm'", name='litellm')
        with patch.object(importlib, 'import_module', side_effect=error):
            with self.assertRaisesRegex(ImportError, r'evoharness\[api\]'):
                APIBackend()

    def test_real_swarm_example_with_mocked_transport(self):
        example = Path(__file__).resolve().parents[1] / 'examples' / 'run_swarm.py'
        with patch.object(sys, 'argv', [str(example), '--model', 'provider/model', '--prompt', 'test task']), \
                patch.object(Path, 'is_file', return_value=True), patch('builtins.print'):
            runpy.run_path(str(example), run_name='__main__')
        self.assertEqual(self.completion.call_count, 4)
        self.load_dotenv.assert_called_once_with(example.parents[1] / '.env', override=False, interpolate=False)


if __name__ == '__main__':
    unittest.main()
