import pandas as pd
import yfinance as yf

class MarketData:
    """Télécharge les prix depuis yfinance"""
    
    def __init__(self, tickers, start_date="2016-01-01"):
        self.tickers = tickers
        self.start_date = start_date

    def load_prices(self):
        price_data = {}
        for asset_name, ticker in self.tickers.items():
            raw = yf.download(
                ticker,
                start=self.start_date,
                auto_adjust=True,
                progress=False,)

            if raw.empty:
                raise ValueError(f"Pas de données pour {asset_name}")

            close = self.extract_close(raw, ticker)
            close.name = asset_name
            price_data[asset_name] = close

        prices = pd.concat(price_data.values(), axis=1)

        prices = prices.sort_index().ffill().dropna()

        return prices

    def extract_close(self, raw, ticker):
        # pour gérer les différentes structures de données de yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            return raw[("Close", ticker)]
        
        return raw["Close"]