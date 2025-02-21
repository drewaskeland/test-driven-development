from SignalDetection import SignalDetection

class Experiment:
    def __init__(self):
        self.conditions_labels_list = []

    def add_condition(self, sig_det_obj: SignalDetection, label: str = None) -> None:
        self.conditions_labels_list.append((sig_det_obj, label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:

    #
    # def compute_auc(self) -> float:
    #
    # def plot_roc_curve(self, show_plot: bool = True):
