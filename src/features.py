"""Agro-climatic feature engineering."""
import pandas as pd


def climate_anomaly(df: pd.DataFrame, col: str, region: str = "region") -> pd.DataFrame:
    baseline = df.groupby(region)[col].transform("mean")
    df[f"{col}_anomaly"] = df[col] - baseline
    return df


def rolling_yield(df: pd.DataFrame, col: str, window: int = 3) -> pd.DataFrame:
    df[f"{col}_roll{window}"] = (
        df.groupby("region")[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    return df
