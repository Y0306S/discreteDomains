import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import queue
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE/'staged/services'), str(HERE/'staged/code'),
               str(ROOT/'services'), str(ROOT/'code')]
from prompt_kv_cache import PromptKVCache, tensor_bytes


class CacheChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32,
            intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=8, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=4, eos_token_id=None, pad_token_id=0,
            rope_parameters={'rope_type':'default','rope_theta':10000.,
                             'partial_rotary_factor':1.0})
        cls.model = Qwen3_5ForCausalLM(config).eval()
        cls.inputs = dict(input_ids=torch.tensor([[3,4,5,6,7,8]]),
            attention_mask=torch.ones(1,6,dtype=torch.long), max_new_tokens=5,
            logits_to_keep=1, do_sample=False, pad_token_id=0)

    def test_hybrid_hit_cold_parity_and_immutable_snapshot(self):
        cache = PromptKVCache(max_bytes=1000000,min_tokens=1)
        with torch.no_grad():
            expected = self.model.generate(**self.inputs)
        cold, first = cache.generate(self.model, 'p', self.inputs)
        hit, second = cache.generate(self.model, 'p', self.inputs)
        self.assertTrue(torch.equal(expected,cold))
        self.assertTrue(torch.equal(cold,hit))
        self.assertFalse(first['hit']);self.assertTrue(second['hit'])
        snapshot,_ = cache.entries['p']
        self.assertEqual(snapshot.get_seq_length(),5)
        self.assertTrue(any(getattr(layer,'recurrent_states',None) is not None
                            for layer in snapshot.layers))

    def test_sampling_restarts_from_prefix_with_fresh_rng(self):
        cache = PromptKVCache(max_bytes=1000000,min_tokens=1)
        kwargs = dict(self.inputs, do_sample=True, top_k=0, temperature=1.0,
                      max_new_tokens=20)
        torch.manual_seed(12);a,_=cache.generate(self.model,'p',kwargs)
        torch.manual_seed(12);b,s=cache.generate(self.model,'p',kwargs)
        torch.manual_seed(13);c,_=cache.generate(self.model,'p',kwargs)
        self.assertTrue(torch.equal(a,b));self.assertTrue(s['hit'])
        self.assertFalse(torch.equal(a,c))

    def test_bounded_eviction_and_identity_isolation(self):
        cache=PromptKVCache(max_bytes=1000000,min_tokens=1)
        cache.generate(self.model,('adapter_a','p'),self.inputs)
        self.assertGreater(cache.bytes,0)
        cache.max_bytes=cache.bytes
        _,s=cache.generate(self.model,('adapter_b','p'),self.inputs)
        self.assertFalse(s['hit']);self.assertEqual(len(cache.entries),1)
        self.assertLessEqual(cache.bytes,cache.max_bytes)

    def test_optional_cache_oom_evicts_and_retries_without_cache(self):
        cache=PromptKVCache(max_bytes=1000000,min_tokens=1)
        with patch.object(cache,'_generate',side_effect=torch.cuda.OutOfMemoryError()):
            result,s=cache.generate(self.model,'p',self.inputs)
        with torch.no_grad():expected=self.model.generate(**self.inputs)
        self.assertTrue(torch.equal(result,expected));self.assertTrue(s['memory_fallback'])
        self.assertEqual(cache.bytes,0)

    def test_query_accumulates_unique_across_identical_prompts_and_partial_fallback(self):
        import batch
        from protocol import canonical
        state={};seen={canonical({'plan':[0]})};calls=[]
        class DB:
            def __enter__(self):return self
            def __exit__(self,*args):pass
        class Tokenizer:
            def apply_chat_template(self,*a,**kw):return 'prompt'
            def __call__(self,*a,**kw):return {'input_ids':[1]}
        class Client:
            state={};last_response={}
            def complete_messages(self,messages,**kw):
                calls.append(copy.deepcopy(messages))
                # Four tries yield only six unique items, keeping all six.
                items=([1,2,3,1,0,2,1,2,3,0] if len(calls)==1
                       else [3,4,5,6,4,5,6,3,0,1])
                return canonical({str(i+1):[v] for i,v in enumerate(items)})
        recipe=dict(batch_policy={'size':10,'max_requests':4,'max_input_tokens':1000,
            'max_new_tokens':{'query':1000}},batch_policy_sha256='batch',
            generation={},seed=42,fallback={'enabled':True},
            query_cache_policy={'reuse_retry_prompt':True},query_cache_policy_sha256='reuse')
        fallback_calls=[]
        def fallback(*a,**kw):
            fallback_calls.append(kw['number'])
            return {'plan':[100+kw['number']]},{'source':'random_fallback'}
        initial=[dict(configuration={'plan':[0]},objective=1,status='success')]
        with patch.object(batch,'choose_fallback',side_effect=fallback):
            result=batch.ensure_batch(DB(),lambda db,k,default=None:state.get(k,default),
                lambda db,k,v:state.__setitem__(k,v),recipe=recipe,domain='query',
                context={},initial=initial,rows=[],total=1000,validate=lambda c:None,
                seen=seen,tokenizer=Tokenizer(),client=Client())
        self.assertEqual(len(calls),4)
        self.assertTrue(all(m==calls[0] for m in calls))
        self.assertEqual([x['answer']['next_configuration']['plan'][0]
                          for x in result['items'][:6]],[1,2,3,4,5,6])
        self.assertEqual(len(fallback_calls),4)
        self.assertEqual(len({canonical(x['answer']['next_configuration'])
                              for x in result['items']}),10)


if __name__=='__main__':unittest.main(verbosity=2)
