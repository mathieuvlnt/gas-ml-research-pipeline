import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

        elif model_name == "Gradient Boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=42)

        elif model_name == "Logistic Regression":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight="balanced"))])

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
        if self.model_name in ["Random Forest", "Gradient Boosting"] :
            importance = self.model.feature_importances_
        else:
            importance = abs(self.model.named_steps["classifier"].coef_[0])
        
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
