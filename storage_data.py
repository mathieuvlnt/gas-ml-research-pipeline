import pandas as pd


class StorageData:
    """Charge les données de stockage gaz européen depuis AGSI+"""

    def __init__(self, filepath="agsi_storage.csv"):
        self.filepath = filepath

    def load_storage(self):

        # Détection automatique du séparateur CSV
        df = pd.read_csv(
            self.filepath,
            sep=None,
            engine="python")

        df.columns = df.columns.str.strip()

        # Recherche des colonnes même si les unités sont présentes dans le nom
        date_col = next(
            c for c in df.columns
            if c.lower().startswith("gas day end"))

        full_col = next(
            c for c in df.columns
            if c.lower().startswith("full"))

        injection_col = next(
            c for c in df.columns
            if c.lower().startswith("injection")
            and "capacity" not in c.lower())

        withdrawal_col = next(
            c for c in df.columns
            if c.lower().startswith("withdrawal")
            and "capacity" not in c.lower())

        # Dates européennes : jour/mois/année
        df[date_col] = pd.to_datetime(
            df[date_col],
            format="mixed",
            dayfirst=True,
            errors="coerce")

        df = (df
              .set_index(date_col)
              .sort_index())

        # Conversion numérique
        storage_full = pd.to_numeric(
            df[full_col],
            errors="coerce")

        injection = pd.to_numeric(
            df[injection_col],
            errors="coerce")

        withdrawal = pd.to_numeric(
            df[withdrawal_col],
            errors="coerce")

        result = pd.DataFrame(index=df.index)

        result["Storage Full"] = storage_full

        result["Storage 7-Day Change"] = (
            storage_full.diff(7))

        result["Net Injection"] = (
            injection - withdrawal)

        # Décalage conservateur pour éviter d'utiliser une donnée de stockage non encore disponible
        result = result.shift(1)

        return result.dropna()