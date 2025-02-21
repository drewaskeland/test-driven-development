import numpy as np
import matplotlib.pyplot as plt
from signal_detection import SignalDetection  # Assuming SignalDetection class is already defined

class Experiment:
    def __init__(self):
        """Initializes an empty experiment with a list of conditions."""
        self.conditions = []  # Stores (SignalDetection object, label) tuples

    def add_condition(self, sdt_obj: SignalDetection, label: str = None) -> None:
        """Add a SignalDetection object with an optional label."""
        self.conditions.append((sdt_obj, label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        """Return sorted false alarm rates and hit rates for ROC curve plotting."""
        if not self.conditions:
            raise ValueError("No conditions have been added to the experiment.")

        false_alarm_rates = [sdt_obj.fa_rate() for sdt_obj, _ in self.conditions]
        hit_rates = [sdt_obj.hit_rate() for sdt_obj, _ in self.conditions]

        # Sort by false alarm rates
        sorted_indices = np.argsort(false_alarm_rates)
        sorted_false_alarm_rates = [false_alarm_rates[i] for i in sorted_indices]
        sorted_hit_rates = [hit_rates[i] for i in sorted_indices]

        return sorted_false_alarm_rates, sorted_hit_rates

    def compute_auc(self) -> float:
        """Compute the Area Under the Curve (AUC) using the trapezoidal rule."""
        if not self.conditions:
            raise ValueError("No conditions have been added to the experiment.")

        false_alarm_rates, hit_rates = self.sorted_roc_points()
        auc = np.trapz(hit_rates, false_alarm_rates)  # Trapezoidal rule
        return auc

    def plot_roc_curve(self, show_plot: bool = True) -> None:
        """Plot the ROC curve for the experiment."""
        false_alarm_rates, hit_rates = self.sorted_roc_points()

        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker="o", linestyle="-", label="ROC Curve")
        plt.plot([0, 1], [0, 1], "k--", label="Chance Level (AUC=0.5)")
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()

        if show_plot:
            plt.show()
