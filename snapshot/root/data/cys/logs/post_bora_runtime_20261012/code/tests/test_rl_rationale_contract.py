import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from examples.benchmark.p5_response_format import format_response
from examples.benchmark.run_p5_rl_bo_prefix_b import sample_one
from examples.benchmark.rl_reward_utils import RewardWeights
from mindware.components.utils.constants import SUCCESS


class RLRationaleContractTests(unittest.TestCase):
    def test_missing_rationale_uses_existing_parse_penalty_and_keeps_result(self):
        config = {'adaboost:learning_rate': 0.1}
        valid = format_response('A smaller step can require more iterations.', config)
        invalid = json.dumps({'config': config})
        for text, expected_penalty in ((valid, 0.0), (invalid, -0.5)):
            with self.subTest(text=text):
                opt = Mock()
                opt._request_config_attempt.return_value = (config, text, {'config': config}, 'note', 0, {})
                opt._is_duplicate.return_value = False
                opt.objective_function.return_value = SimpleNamespace(objectives=[-0.7], trial_state=SUCCESS)
                args = SimpleNamespace(llm_response_format='rationale_json', dataset='test', task_type='cls',
                                       llm_temperature=1.0)
                result = sample_one(opt=opt, args=args, sample_i=0, k=3, group_id='group', scenario='normal',
                    prompt_id='P5', variant_name='test', run_id='test', algo='adaboost', bo_path=Path('history.json'),
                    messages=[{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'user'}],
                    history_meta={}, prior_perfs=[0.5, 0.6], weights=RewardWeights())
                self.assertEqual(result['completion'], text)
                self.assertEqual(result['reward']['perf'], 0.7)
                self.assertEqual(result['reward']['reward_components']['parse_fail'], expected_penalty)
                self.assertEqual(result['response_format_valid'], expected_penalty == 0)
                opt.objective_function.assert_called_once_with(config)
                opt._sample_nonduplicate_config.assert_not_called()


if __name__ == '__main__':
    unittest.main()
