import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix)


class ModelEvaluator:
    """Calcule métriques de performance du modèle"""

    def __init__(self, y_true, y_pred, target_type="direction"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.target_type = target_type

    def get_metrics(self):

        return {
            "Accuracy": accuracy_score(self.y_true, self.y_pred),
            "Balanced Accuracy": balanced_accuracy_score(self.y_true, self.y_pred),
            "Precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "Recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "F1 Score": f1_score(
                self.y_true, self.y_pred, zero_division=0)}

    def get_confusion_matrix(self):
        matrix = confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])
        if self.target_type in ["direction", "direction_5d"]:
            labels = ["Down", "Up"]
        elif self.target_type == "large_move":
            labels = ["No Large Move", "Large Move"]
        elif self.target_type == "high_volatility":
            labels = ["Normal Volatility", "High Volatility"]
        else:
            labels = ["Class 0", "Class 1"]

            
        return pd.DataFrame(
            matrix,
            index=[f"Actual {labels[0]}",
                   f"Actual {labels[1]}"],
            columns=[f"Predicted {labels[0]}",
                     f"Predicted {labels[1]}"])