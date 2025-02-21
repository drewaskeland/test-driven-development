import unittest
from your_module import Experiment, SignalDetection

class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.exp = Experiment()

    def test_add_condition(self):
        sdt_obj = SignalDetection(40, 10, 20, 30)
        self.exp.add_condition(sdt_obj, label="Condition A")
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertEqual(self.exp.conditions[0][1], "Condition A")

    def test_sorted_roc_points(self):
        self.exp.add_condition(SignalDetection(40, 10, 20, 30), label="Condition A")
        self.exp.add_condition(SignalDetection(30, 20, 15, 35), label="Condition B")
        false_alarm_rate, hit_rate = self.exp.sorted_roc_points()
        self.assertEqual(sorted(false_alarm_rate), false_alarm_rate)
        self.assertEqual(len(false_alarm_rate), len(hit_rate))

    def test_compute_auc_known_cases(self):
        # Test case for AUC = 0.5
        self.exp.add_condition(SignalDetection(0, 1, 1, 0))
        self.assertEqual(self.exp.compute_auc(), 0.5)

        # Test case for AUC = 1
        self.exp.add_condition(SignalDetection(0, 1, 1, 0))
        self.exp.add_condition(SignalDetection(0, 1, 0, 1))
        self.assertEqual(self.exp.compute_auc(), 1.0)

    def test_empty_experiment(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()

        with self.assertRaises(ValueError):
            self.exp.compute_auc()

if __name__ == '__main__':
    unittest.main()
