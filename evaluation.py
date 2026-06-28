import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix)


class ModelEvaluator:
    """Calcule métrique de performance du modèle"""

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def get_metrics(self):

        return {
            "Accuracy": accuracy_score(self.y_true, self.y_pred),
            "Precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "Recall": recall_score(self.y_true, self.y_pred, zero_division=0)}

    def get_confusion_matrix(self):
        matrix = confusion_matrix(self.y_true, self.y_pred)
        return pd.DataFrame(
            matrix,
            index=["Actual Down", "Actual Up"],
            columns=["Predicted Down", "Predicted Up"])
