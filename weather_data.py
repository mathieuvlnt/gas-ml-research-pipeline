import time
import requests
import pandas as pd


class WeatherData:
    """Télécharge les HDD européens"""

    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def load_hdd(self):

        cities = {
            "Amsterdam": (52.37, 4.90),
            "Berlin": (52.52, 13.41),
            "Paris": (48.86, 2.35),
            "London": (51.51, -0.13)}

        all_hdd = []

        for city, (latitude, longitude) in cities.items():

            url = "https://archive-api.open-meteo.com/v1/archive"

            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "daily": "temperature_2m_mean",
                "timezone": "Europe/Paris"}

            for attempt in range(3):

                try:
                    response = requests.get(
                        url,
                        params=params,
                        timeout=60)

                    response.raise_for_status()
                    data = response.json()
                    break

                except requests.RequestException:

                    if attempt == 2:
                        raise

                    time.sleep(2)

            temperature = pd.Series(
                data["daily"]["temperature_2m_mean"],
                index=pd.to_datetime(
                    data["daily"]["time"]),
                name=city)

            hdd = (
                18 - temperature
            ).clip(lower=0)

            all_hdd.append(hdd)

        hdd_data = pd.concat(
            all_hdd,
            axis=1)

        european_hdd = hdd_data.mean(axis=1)

        result = pd.DataFrame(
            index=european_hdd.index)

        result["European HDD"] = european_hdd

        result["HDD 7-Day Average"] = (
            european_hdd
            .rolling(7)
            .mean())

        return result.dropna()