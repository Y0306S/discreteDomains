import threading
import unittest
from unittest.mock import patch

import qwen_inference_server as server
from qwen_inference_transport import SharedInferenceError, digest


class ContextRoutingTests(unittest.TestCase):
    def replica(self):
        obj = server.Replica.__new__(server.Replica)
        obj.name = 'shared_a16'
        obj.config = dict(context_fallback_urls=['http://a', 'http://b'],
                          max_input_tokens=16384, max_new_tokens=1024,
                          fallback_max_input_tokens=32768)
        obj.forwarded = obj.completed = 0
        obj.lock = threading.Lock()
        obj.failed = None
        obj.identities = [dict(adapter='pinned')]
        return obj

    def body(self):
        body = dict(identity=dict(adapter='pinned'), messages=[dict(role='user', content='whole prompt')],
                    settings=dict(temperature=1., max_new_tokens=1024, top_p=.9,
                                  top_k=50, enable_thinking=False),
                    rng=dict(seed=44, cuda_rng='saved'), use_s1b_template=True)
        body['request_id'] = digest(body)
        return body

    def test_forward_preserves_request_and_sampling_result(self):
        obj = self.replica(); body = self.body()
        response = dict(identity=body['identity'], text='answer', cuda_rng='next', replica='full')
        calls = []
        def http(url, payload=None, timeout=None):
            calls.append((url, payload))
            return dict(ready=True, busy=False, identities=obj.identities) if url.endswith('/health') else response
        with patch.object(server, 'http_json', side_effect=http):
            result = obj.forward_long_context(body, 20000)
        self.assertIs(calls[-1][1], body)
        for key in response:self.assertEqual(result[key], response[key])
        self.assertEqual(obj.completed, 1)
        self.assertEqual(obj.forwarded, 1)

    def test_wrong_identity_is_never_selected(self):
        obj = self.replica(); calls = []
        def http(url, payload=None, timeout=None):
            calls.append(url)
            if url == 'http://a/health':return dict(ready=True, identities=[dict(adapter='wrong')])
            if url.endswith('/health'):return dict(ready=True, busy=False, identities=obj.identities)
            return dict(text='answer')
        with patch.object(server, 'http_json', side_effect=http):obj.forward_long_context(self.body(), 20000)
        self.assertEqual(calls[-1], 'http://b/generate')

    def test_unavailable_full_context_pool_fails_without_generation(self):
        obj = self.replica()
        with patch.object(server, 'http_json', return_value=dict(ready=False)) as http:
            with self.assertRaises(SharedInferenceError):obj.forward_long_context(self.body(), 20000)
        self.assertTrue(all(call.args[0].endswith('/health') for call in http.call_args_list))
        self.assertEqual(obj.completed, 0)

    def test_oversize_prompt_is_rejected_without_truncation(self):
        obj = self.replica(); obj.input_tokens = lambda body: 32769
        with patch.object(obj, 'forward_long_context') as forward:
            with self.assertRaisesRegex(SharedInferenceError, 'no truncation'):obj.generate(self.body())
        forward.assert_not_called()

    def test_long_prompt_routes_before_gpu_generation(self):
        obj = self.replica(); body = self.body(); obj.input_tokens = lambda body: 16385
        with patch.object(obj, 'forward_long_context', return_value=dict(text='answer')) as forward:
            self.assertEqual(obj.generate(body), dict(text='answer'))
        forward.assert_called_once_with(body, 16385)


if __name__ == '__main__':unittest.main()
