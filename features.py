import numpy as np
import pandas as pd

class FeatureEngine:
    """Calcule les indicateurs quantitatifs"""

    def __init__(self, prices):
        self.prices = prices
        
    def compute_features(self):
        features = pd.DataFrame(index=self.prices.index)
        # Variables saisonnières
        features["Winter"] = features.index.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
        features["Month Sin"] = np.sin(2 * np.pi * features.index.month / 12)
        features["Month Cos"] = np.cos(2 * np.pi * features.index.month / 12)
        
        # Calcul des rdts logarithmiques
        returns = np.log(self.prices / self.prices.shift(1))

        features["Return"] = returns["TTF"]

        # Rendements retardés
        features["Return Lag 1"] = returns["TTF"].shift(1)
        features["Return Lag 2"] = returns["TTF"].shift(2)
        features["Return Lag 5"] = returns["TTF"].shift(5)
        
        # Volatilité à différents horizons
        features["5-Day Volatility"] = (returns["TTF"]
                                        .rolling(5)
                                        .std()
                                        * np.sqrt(252))
        features["10-Day Volatility"] = (returns["TTF"]
                                         .rolling(10)
                                         .std()
                                         * np.sqrt(252))
        features["60-Day Volatility"] = (returns["TTF"]
                                         .rolling(60)
                                         .std()
                                         * np.sqrt(252))

        features["10-Day Momentum"] = self.prices["TTF"].pct_change(10)
        features["30-Day Momentum"] = self.prices["TTF"].pct_change(30)

        features["20-Day Volatility"] = (returns["TTF"]
                                        .rolling(20)
                                        .std()
                                        * np.sqrt(252))

        if "BRENT" in self.prices.columns:
            features["Brent Return"] = returns["BRENT"]
            features["Brent 5-Day Momentum"] = self.prices["BRENT"].pct_change(5)
            features["Brent 20-Day Momentum"] = self.prices["BRENT"].pct_change(20)
            features["Correlation"] = (
                returns["TTF"]
                .rolling(30)
                .corr(returns["BRENT"]))

            spread = self.prices["TTF"] / self.prices["BRENT"]
            features["Spread"] = spread
            features["Spread Z-Score"] = self.compute_zscore(spread, 30)

        features["TTF Z-Score"] = self.compute_zscore(
            self.prices["TTF"], 30)

        features["Volatility Regime"] = self.compute_volatility_regime(
            features["20-Day Volatility"])

        return features.replace([np.inf, -np.inf], np.nan).dropna()

    def compute_zscore(self, series, window):
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std

    def compute_volatility_regime(self, volatility):
        # Seuils calculés uniquement à partir des données passées
        historical_volatility = volatility.shift(1)
        
        low = historical_volatility.expanding(min_periods=60).quantile(0.33)
        high = historical_volatility.expanding(min_periods=60).quantile(0.66)

        regime = pd.Series(np.nan, index=volatility.index)
        valid = volatility.notna() & low.notna() & high.notna()
        
        regime.loc[valid] = 1
        regime.loc[valid & (volatility < low)] = 0
        regime.loc[valid & (volatility > high)] = 2
    
        return regime