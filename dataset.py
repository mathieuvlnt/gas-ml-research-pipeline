import pandas as pd

class DatasetBuilder:
    """Construit le jeu de données pour le ML"""
    def __init__(self, features, prices, target_type="direction", threshold=0.02):
        self.features = features
        self.prices = prices
        self.target_asset = "TTF"
        self.target_type = target_type
        self.threshold = threshold
        self.test_size = 0.20

    def build_dataset(self):
        """Construit les jeux d'entraînement et de test"""
        # Construction de la variable cible
        future_return = self.prices[self.target_asset].pct_change().shift(-1)
        
        if self.target_type == "direction":
            target = (future_return > 0).astype(int)
        elif self.target_type == "large_move":
            target = (abs(future_return) > self.threshold).astype(int)
        elif self.target_type == "high_volatility":
            future_volatility = (
                self.prices[self.target_asset]
                .pct_change()
                .rolling(5)
                .std()
                .shift(-5)
            )
            
            threshold = future_volatility.quantile(0.75)
            target = (future_volatility > threshold).astype(int)
        else:
            raise ValueError("Target inconnue")
        target = target.rename("Target")

        # Création des variables explicatives (X) et de la cible (y)
        dataset = self.features.join(target).dropna()
        X = dataset.drop(columns=["Target"])
        y = dataset["Target"]

        # Séparation chronologique pour avoir 80% entrainement/20% test
        split_index = int(len(dataset) * (1 - self.test_size))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        return X_train, X_test, y_train, y_test