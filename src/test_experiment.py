import unittest
from experiment import Experiment
from signal_detection import SignalDetection

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
    def test_compute_auc_known_cases(self):
        # Create signal detection objects
        sd1 = SignalDetection(5, 2, 8, 2)  # hit_rate = 0.714, fa_rate = 0.8
        sd2 = SignalDetection(3, 1, 2, 1)   # hit_rate = 0.75, fa_rate = 0.67
    
        # Add them to the experiment
        self.exp.add_condition(sd1, "Low")
        self.exp.add_condition(sd2, "High")

        auc = self.exp.compute_auc()
        print(f"Computed AUC: {auc}")  # Should print AUC value

        self.assertEqual(auc, 0.5)  # If AUC = 0.5 for these specific cases

if __name__ == '__main__':
    unittest.main()
