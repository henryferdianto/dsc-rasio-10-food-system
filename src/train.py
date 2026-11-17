"""Train a cross-region food system resilience model."""
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", default="resilience_index")
    ap.add_argument("--group", default="region")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    y = df[args.target]
    X = pd.get_dummies(df.drop(columns=[args.target]), drop_first=True)

    cv = GroupKFold(n_splits=5)
    scores = []
    for tr, te in cv.split(X, y, groups=df[args.group]):
        model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        scores.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
    print(f"CV RMSE: {np.mean(scores):.4f}")
    model.fit(X, y)
    joblib.dump(model, "models/rasio.joblib")


if __name__ == "__main__":
    main()
