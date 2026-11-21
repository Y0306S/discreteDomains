import copy
import json
import unittest
from peptide_prompt import valid_first_occurrences
from rl_prompt_policy import build_peptide_rl_prompt, retrieval_evidence
from test_pairs import REF, observation, variants


def peptide_task(task_id, reference):
    return dict(task_id=task_id, description=f'Optimize antimicrobial peptide relative to reference {reference}',
        objective='minimize APEX bacteria_0 MIC', reference_sequence=reference,
        constraints=dict(alphabet='ACDEFGHIKLMNPQRSTVWY', min_length=1,max_length=30,max_cysteine=1,
                         min_reference_similarity=.75,similarity_metric='1-Levenshtein(candidate,reference)/len(reference)'))


class RLPolicyTests(unittest.TestCase):
    def setUp(self):
        self.current=peptide_task('target',REF)
        self.source=peptide_task('source','GKLRAKLR')
        self.card=dict(task=self.source,top=[dict(candidate='GGLRAKLR',measured_value=10.)],
                       bottom=[dict(candidate='GALRAKLR',measured_value=80.)])
        _,self.prefix=valid_first_occurrences([observation(s,100+i) for i,s in enumerate(variants(60))],REF)

    def test_rs_has_real_descriptions_and_differences(self):
        messages,c=build_peptide_rl_prompt(self.current,self.prefix,scenario='retrieval',round_number=4,
                                          cards=[self.card],allowed_training_tasks={'source'})
        evidence=c['similar_task_evidence'];card=evidence['similar_tasks'][0]
        self.assertEqual(card['similar_task'],self.source)
        diff=next(x for x in card['differences_from_current_task'] if x['field']=='reference_sequence')
        self.assertFalse(diff['same']);self.assertEqual(diff['current_task_value'],REF)
        self.assertEqual(diff['similar_task_value'],'GKLRAKLR')
        self.assertEqual(len(c['recent50']),50)
        self.assertEqual(len(c['good']),5);self.assertEqual(len(c['bad']),5)
        self.assertEqual(json.loads(messages[1]['content']),c)
        self.assertEqual(set(card),{'similar_task','differences_from_current_task','top','bottom','score_scope'})

    def test_post50_exploitation_requires_evidence(self):
        for scenario,round_number in [('retrieval',4),('exploit',51)]:
            with self.assertRaises(ValueError):
                build_peptide_rl_prompt(self.current,self.prefix,scenario=scenario,round_number=round_number)
        for scenario,round_number in [('normal',30),('explore',70),('exploit',50)]:
            _,c=build_peptide_rl_prompt(self.current,self.prefix,scenario=scenario,round_number=round_number)
            self.assertNotIn('similar_task_evidence',c)
        _,c=build_peptide_rl_prompt(self.current,self.prefix,scenario='exploit',round_number=51,
                                   cards=[self.card],allowed_training_tasks={'source'})
        self.assertIn('similar_task_evidence',c)

    def test_regions_missing_metadata_and_heldout_fail_closed(self):
        for key in ('good_region','bad_region','latent_bounds'):
            card=copy.deepcopy(self.card);card[key]={'x':[0,1]}
            with self.assertRaises(ValueError):retrieval_evidence('peptide',self.current,[card],{'source'})
        with self.assertRaises(ValueError):retrieval_evidence('peptide',self.current,[self.card],{'other'})
        card=copy.deepcopy(self.card);card['task']['description']=''
        with self.assertRaises(ValueError):retrieval_evidence('peptide',self.current,[card],{'source'})
        card=copy.deepcopy(self.card);card['task']['task_id']='target'
        with self.assertRaises(ValueError):retrieval_evidence('peptide',self.current,[card],{'target'})
        card=copy.deepcopy(self.card);card['top'][0]['measured_value']=float('nan')
        with self.assertRaises(ValueError):retrieval_evidence('peptide',self.current,[card],{'source'})

    def test_query_uses_query_text_and_discrete_space_differences(self):
        target=dict(task_id='q1',description='Current count query',objective='minimize measured runtime',
                    constraints={'valid_plan':True},query_text='SELECT COUNT(*) FROM target_table',
                    search_space={'join_order':['a-b','b-a']})
        source=dict(task_id='q2',description='Similar filtered query',objective='minimize measured runtime',
                    constraints={'valid_plan':True},query_text='SELECT COUNT(*) FROM source_table WHERE x > 5',
                    search_space={'join_order':['b-a']})
        card=dict(task=source,top=[dict(candidate={'join_order':'b-a'},measured_value=1.)],
                  bottom=[dict(candidate={'join_order':'b-a'},measured_value=2.)])
        evidence=retrieval_evidence('query',target,[card],{'q2'})
        diff={x['field']:x for x in evidence['similar_tasks'][0]['differences_from_current_task']}
        self.assertFalse(diff['query_text']['same']);self.assertFalse(diff['search_space']['same'])
        self.assertEqual(evidence['current_task']['query_text'],target['query_text'])
        # Caller mutation cannot inject region fields into assembled evidence.
        card['bad_region']={'x':[0,1]}
        self.assertNotIn('bad_region',evidence['similar_tasks'][0])


if __name__=='__main__':unittest.main()
