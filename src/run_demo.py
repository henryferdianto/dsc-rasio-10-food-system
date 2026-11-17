import pandas as pd

from src.sample_data import make
from src.train import train


def main():
    df = make()
    df.to_csv("data/raw/food_system.csv", index=False)
    _, _, cv_rmse, metrics = train(df)
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
