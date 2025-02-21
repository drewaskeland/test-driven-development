import unittest
from signal_detection import SignalDetection
from experiment import Experiment  # Assuming Experiment is in experiment.py

class TestExperiment(unittest.TestCase):

    def setUp(self):
        """Create a new Experiment instance before each test."""
        self.exp = Experiment()

    def test_add_condition(self):
        """Test adding a condition."""
        sd1 = SignalDetection(5, 2, 8, 2)
        self.exp.add_condition(sd1, "Condition A")
        self.assertEqual(len(self.exp.conditions), 1)

    def test_sorted_roc_points(self):
        """Test that sorted_roc_points returns sorted false alarm rates and hit rates."""
        sd1 = SignalDetection(5, 2, 8, 2)  # hit_rate = 0.714, fa_rate = 0.8
        sd2 = SignalDetection(3, 1, 2, 1)  # hit_rate = 0.75, fa_rate = 0.67
        self.exp.add_condition(sd1, "Condition A")
        self.exp.add_condition(sd2, "Condition B")
        
        false_alarm_rate, hit_rate = self.exp.sorted_roc_points()
        
        self.assertEqual(len(false_alarm_rate), 2)
        self.assertEqual(len(hit_rate), 2)
        self.assertGreater(false_alarm_rate[1], false_alarm_rate[0])  # Check that fa_rate is sorted

    def test_compute_auc(self):
        """Test computing AUC."""
        sd1 = SignalDetection(5, 2, 8, 2)  # hit_rate = 0.714, fa_rate = 0.8
        sd2 = SignalDetection(3, 1, 2, 1)  # hit_rate = 0.75, fa_rate = 0.67
        self.exp.add_condition(sd1, "Low Noise")
        self.exp.add_condition(sd2, "High Noise")

        auc = self.exp.compute_auc()
        print(f"Computed AUC: {auc}")
        self.assertAlmostEqual(auc, 0.0976, places=3)  # Test with expected AUC value

    def test_empty_experiment(self):
        """Test that an error is raised when no conditions are added."""
        with self.assertRaises(ValueError):
            self.exp.compute_auc()

    def test_auc_two_conditions(self):
        """Test AUC when two conditions are perfectly diagonal."""
        sd1 = SignalDetection(5, 5, 5, 5)  # hit_rate = 0.5, fa_rate = 0.5
        sd2 = SignalDetection(10, 0, 0, 10)  # hit_rate = 1.0, fa_rate = 0.0
        self.exp.add_condition(sd1, "Condition A")
        self.exp.add_condition(sd2, "Condition B")

        auc = self.exp.compute_auc()
        print(f"Computed AUC: {auc}")
        self.assertAlmostEqual(auc, 0.5, places=2)

    def test_auc_three_conditions(self):
        """Test AUC when three conditions create a perfect ROC curve."""
        sd1 = SignalDetection(5, 5, 5, 5)  # hit_rate = 0.5, fa_rate = 0.5
        sd2 = SignalDetection(10, 0, 0, 10)  # hit_rate = 1.0, fa_rate = 0.0
        sd3 = SignalDetection(0, 10, 10, 0)  # hit_rate = 0.0, fa_rate = 1.0
        self.exp.add_condition(sd1, "Condition A")
        self.exp.add_condition(sd2, "Condition B")
        self.exp.add_condition(sd3, "Condition C")

        auc = self.exp.compute_auc()
        print(f"Computed AUC: {auc}")
        self.assertEqual(auc, 1.0)

if __name__ == "__main__":
    unittest.main()

