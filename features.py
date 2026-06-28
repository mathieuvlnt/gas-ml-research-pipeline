import numpy as np
import pandas as pd

class FeatureEngine:
    """Calcule les indicateurs quantitatifs"""

    def __init__(self, prices):
        self.prices = prices
        
    def compute_features(self):
        features = pd.DataFrame(index=self.prices.index)
        # Calcul des rdts logarithmiques
        returns = np.log(self.prices / self.prices.shift(1))

        features["Return"] = returns["TTF"]

        features["10-Day Momentum"] = self.prices["TTF"].pct_change(10)
        features["30-Day Momentum"] = self.prices["TTF"].pct_change(30)

        features["Volatility"] = (
            returns["TTF"]
            .rolling(20)
            .std()
            * np.sqrt(252))

        if "BRENT" in self.prices.columns:
            features["Brent Return"] = returns["BRENT"]
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
            features["Volatility"])

        return features.replace([np.inf, -np.inf], np.nan).dropna()

    def compute_zscore(self, series, window):

        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        return (series - rolling_mean) / rolling_std

    def compute_volatility_regime(self, volatility):

        low = volatility.quantile(0.33)
        high = volatility.quantile(0.66)
        
        regime = pd.Series(1, index=volatility.index)
        regime[volatility < low] = 0
        regime[volatility > high] = 2
        return regime