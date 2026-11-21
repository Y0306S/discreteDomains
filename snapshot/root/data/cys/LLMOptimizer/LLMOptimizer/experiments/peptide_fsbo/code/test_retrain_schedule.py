import copy
import json
import unittest

import retrain_schedule as schedule


class RetrainScheduleTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(stagnation_rounds=20, max_retrain_attempts=2)
        self.state = dict(progress=dict(rounds=0, fresh_evaluations=0), rows=[], adaptation_events=[])

    def advance(self, n, phase='acquisition'):
        for _ in range(n):
            self.state['progress']['rounds'] += phase == 'acquisition'
            schedule.observe(self.state, phase, -5., -5.)

    def test_first_attempt_only_after_twenty_flat_rounds(self):
        self.advance(19)
        self.assertFalse(schedule.due(self.state, self.cfg))
        self.advance(1)
        self.assertTrue(schedule.due(self.state, self.cfg))
        i = schedule.reserve(self.state, self.cfg)
        self.assertEqual(self.state['adaptation_events'][i]['round'], 20)
        self.assertEqual(self.state['adaptation_events'][i]['trigger_nonimproving_rounds'], 20)

    def test_any_phase_improvement_resets_patience(self):
        for phase in ('acquisition', 'recenter'):
            self.advance(19)
            schedule.observe(self.state, phase, -5., -4.)
            self.assertEqual(self.state['retrain_schedule']['rounds_without_improvement'], 0)
            self.advance(19)
            self.assertFalse(schedule.due(self.state, self.cfg))
            schedule.observe(self.state, phase, -5., -4.)

    def test_recenter_batches_do_not_count_as_rounds(self):
        self.advance(19)
        self.advance(100, phase='recenter')
        self.assertFalse(schedule.due(self.state, self.cfg))

    def test_both_rejected_and_accepted_attempts_consume_cap(self):
        for accepted in (False, True):
            self.advance(20)
            i = schedule.reserve(self.state, self.cfg)
            self.state['adaptation_events'][i].update(accepted=accepted, status='complete')
            self.assertFalse(schedule.due(self.state, self.cfg))
        self.advance(100)
        self.assertFalse(schedule.due(self.state, self.cfg))
        with self.assertRaises(ValueError):
            schedule.reserve(self.state, self.cfg)
        self.assertEqual(len(self.state['adaptation_events']), 2)

    def test_checkpoint_roundtrip_preserves_counter_and_interrupted_attempt(self):
        self.advance(20)
        schedule.reserve(self.state, self.cfg)
        restored = json.loads(json.dumps(self.state))
        events = schedule.recover(restored)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['status'], 'interrupted_rolled_back')
        self.assertFalse(events[0]['accepted'])
        self.assertEqual(restored['retrain_schedule']['attempts'], 1)
        self.assertEqual(schedule.recover(restored), [])

    def test_no_flat_rounds_after_second_attempt_can_trigger_a_third(self):
        for _ in range(2):
            self.advance(20)
            schedule.reserve(self.state, self.cfg)
        restored = copy.deepcopy(self.state)
        schedule.recover(restored)
        for _ in range(1000):
            schedule.observe(restored, 'acquisition', -5., -5.)
            self.assertFalse(schedule.due(restored, self.cfg))


if __name__ == '__main__':
    unittest.main(verbosity=2)
