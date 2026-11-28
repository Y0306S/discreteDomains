import copy
import os
import unittest
from common import *
from runtime import validate,fleet

class PlacementTests(unittest.TestCase):
    def setUp(self):self.policy=load(ROOT/'placement.json')
    def test_placement(self):validate(self.policy)
    def test_database_exclusivity(self):
        self.policy['workers'][1]['db_port']=self.policy['workers'][0]['db_port']
        with self.assertRaises(RuntimeError):validate(self.policy)
    def test_cpu_exclusivity(self):
        self.policy['workers'][1]['optimizer_cpu']=self.policy['workers'][0]['optimizer_cpu']
        with self.assertRaises(RuntimeError):validate(self.policy)
    def test_recipe_identity(self):
        self.policy['recipe_sha256']='invalid'
        with self.assertRaises(RuntimeError):validate(self.policy)
    def test_process_adoption_uses_actual_slot(self):
        r=dict(pid=os.getpid(),start_ticks=ticks(os.getpid()),slot=14)
        adopted=fleet.AdoptedProcess(r,'bolt')
        self.assertEqual(adopted.number,14)
        self.assertIsNone(adopted.poll())
    def test_new_endpoint_pool_is_independent_of_old_ports(self):
        self.assertEqual(len(set(self.policy['vae_endpoints'])),16)
        self.assertTrue(all(24300<=int(u.rsplit(':',1)[1])<=24315 for u in self.policy['vae_endpoints']))
    def test_second_host_slots_are_disjoint(self):
        from cpu5_runtime import validate as validate_second
        other=load(ROOT/'placement_cpu5.json');validate_second(other)
        self.assertFalse({s['slot'] for s in self.policy['workers']} & {s['slot'] for s in other['workers']})
    def test_second_host_rejects_duplicate_cpu(self):
        from cpu5_runtime import validate as validate_second
        other=load(ROOT/'placement_cpu5.json')
        other['workers'][1]['optimizer_cpu']=other['workers'][0]['optimizer_cpu']
        with self.assertRaises(AssertionError):validate_second(other)

if __name__=='__main__':unittest.main()
