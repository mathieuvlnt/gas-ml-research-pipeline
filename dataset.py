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
            target = (future_return > 0).astype(float)
            target[future_return.isna()] = float("nan")
            horizon=1
        elif self.target_type == "direction_5d":
            future_return_5d = (self.prices[self.target_asset]
                                .pct_change(5)
                                .shift(-5))
            target = (future_return_5d > 0).astype(float)
            target[future_return_5d.isna()] = float("nan")
            horizon = 5
        elif self.target_type == "large_move":
            target = (abs(future_return) > self.threshold).astype(float)
            target[future_return.isna()] = float("nan")
            horizon = 1
        elif self.target_type == "high_volatility":
            future_volatility = (self.prices[self.target_asset]
                                 .pct_change()
                                 .rolling(5)
                                 .std()
                                 .shift(-5))

            # Alignement avec fetaures pr déterminer partie train
            temp_dataset = self.features.join(future_volatility.rename("FutureVolatility")).dropna()
            temp_split_index = int(len(temp_dataset) * (1 - self.test_size))

            # Purge des 5 dernières observations du train
            temp_train_end = temp_split_index - 5

            # Seuil calculé uniquement sur train
            threshold = (temp_dataset["FutureVolatility"].iloc[:temp_train_end].quantile(0.75))
            
            target = (future_volatility > threshold).astype(float)
            target[future_volatility.isna()] = float("nan")
            horizon = 5
        else:
            raise ValueError("Target inconnue")

        target = target.rename("Target")
        
        # Création des variables explicatives (X) et de la cible (y)
        dataset = self.features.join(target).dropna()
        X = dataset.drop(columns=["Target"])
        y = dataset["Target"]

        # Séparation chronologique pour avoir 80% entrainement/20% test
        split_index = int(len(dataset) * (1 - self.test_size))

        # Evite que les targets du train chevauchent la période test
        train_end = split_index - horizon
        
        X_train = X.iloc[:train_end]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:train_end]
        y_test = y.iloc[split_index:]

        return X_train, X_test, y_train, y_test