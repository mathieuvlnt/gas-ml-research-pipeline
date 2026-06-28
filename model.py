import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class ModelTrainer:

    def __init__(self, model_name="Random Forest"):

        self.model_name = model_name
        # Sélection du modèle de ML
        if model_name == "Random Forest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                random_state=42,
                class_weight="balanced")

        elif model_name == "Logistic Regression":
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000)

        else:
            raise ValueError("Unknown model")
            
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        predictions = self.model.predict(X_test)
        return pd.Series(predictions, index=X_test.index, name="Prediction")

    def predict_proba(self, X_test):
        probabilities = self.model.predict_proba(X_test)[:, 1]
        return pd.Series(probabilities, index=X_test.index, name="Probability")

    def get_feature_importance(self, feature_names):
        # Calcul de l'importance des variables
        if self.model_name == "Random Forest":
            importance = self.model.feature_importances_
        else:
            importance = abs(self.model.coef_[0])
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
