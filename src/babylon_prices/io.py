import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_excel(path, index=False)
