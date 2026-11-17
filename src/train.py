"""Train a cross-region food system resilience model."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from src.features import build


def train(df: pd.DataFrame, target: str = "resilience_index", group: str = "region"):
    y = df[target]
    X = build(df.drop(columns=[target]))
    cv = GroupKFold(n_splits=5)
    rmses = []
    for tr, te in cv.split(X, y, groups=df[group]):
        model = HistGradientBoostingRegressor(random_state=42)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
    model = HistGradientBoostingRegressor(random_state=42).fit(X, y)
    in_sample = model.predict(X)
    return model, list(X.columns), float(np.mean(rmses)), {
        "mae": float(mean_absolute_error(y, in_sample)),
        "rmse": float(np.sqrt(mean_squared_error(y, in_sample))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/food_system.csv")
    args = ap.parse_args()
    if not Path(args.input).exists():
        from src.sample_data import main as make_sample

        make_sample()
    model, columns, cv_rmse, metrics = train(pd.read_csv(args.input))
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": columns}, "models/rasio.joblib")
    print(f"GroupKFold RMSE={cv_rmse:.4f} | in-sample {metrics}")


if __name__ == "__main__":
    main()
