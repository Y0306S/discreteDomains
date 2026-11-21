import unittest

from handoff import choose_cpus, completed, held_task_cpus


class HandoverTests(unittest.TestCase):
    def test_final_fit_still_holds_its_cores(self):
        self.assertEqual(held_task_cpus({'cpus': [1, 2, 3]},
            {'attempts': 100, 'budget_exhausted': True, 'pending': False}, True), {1, 2, 3})

    def test_interrupted_task_keeps_its_cpu_reservation(self):
        self.assertEqual(held_task_cpus({'cpus': [1, 2, 3]},
            {'attempts': 80, 'budget_exhausted': False}, False), {1, 2, 3})

    def test_completed_exited_worker_releases_cores(self):
        self.assertEqual(held_task_cpus({'cpus': [1, 2, 3]},
            {'attempts': 100, 'budget_exhausted': True, 'pending': False}, False), set())

    def test_pending_evaluation_prevents_completion(self):
        self.assertFalse(completed({'attempts': 100, 'budget_exhausted': True, 'pending': True}))

    def test_siblings_and_existing_jobs_are_excluded(self):
        key = lambda cpu: cpu % 8
        self.assertEqual(choose_cpus(list(range(16)), {0, 1}, {2}, {3}, key), [4, 5, 6, 7])

    def test_fewer_than_four_distinct_free_cores_waits(self):
        self.assertIsNone(choose_cpus([0, 1, 2, 8, 9, 10], set(), set(), set(), lambda cpu: cpu % 8))


if __name__ == '__main__':
    unittest.main()
