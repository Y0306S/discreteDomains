"""CPU transport for exact rollout token records; no implicit retries."""
from rl_common import ROOT
from mindware.domains.dlhpo.taskset_llm.inference import SharedQwenClient, InferenceServiceError


class ExactRolloutClient(SharedQwenClient):
    def generate(self, messages, *, n=1, temperature=1., seed=None):
        result = self._request('/generate', dict(identity_sha256=self.identity_sha256,
            messages=messages, n=n, temperature=temperature, seed=seed))
        replies = result.get('replies')
        if (result.get('identity_sha256') != self.identity_sha256 or not isinstance(replies, list)
                or len(replies) != n):
            raise InferenceServiceError('Invalid exact-token response identity/count')
        for row in replies:
            if (not isinstance(row, dict) or not isinstance(row.get('completion'), str)
                    or any(not isinstance(row.get(k), list) or not row[k]
                           or any(type(t) is not int or t < 0 for t in row[k])
                           for k in ('prompt_ids', 'answer_ids'))):
                raise InferenceServiceError('Missing exact sampled token sequence')
        return replies
