"""Generate synthetic agro-climatic panel data for resilience modelling."""
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["Indonesia", "Malaysia", "Thailand", "Vietnam", "Philippines"]


def make(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for region_idx, region in enumerate(REGIONS):
        for year in range(2010, 2025):
            rainfall = rng.normal(2200, 250)
            temp_anomaly = rng.normal(0.4, 0.3)
            gdp = 4000 + region_idx * 900 + (year - 2010) * 180 + rng.normal(0, 200)
            yield_t = 3.2 + (year - 2010) * 0.03 - 0.2 * temp_anomaly + rng.normal(0, 0.15)
            resilience = (
                0.35 * (yield_t / 3.5)
                + 0.3 * (gdp / 9000)
                + 0.2 * (rainfall / 2500)
                - 0.4 * temp_anomaly
                + rng.normal(0, 0.05)
            )
            rows.append({
                "region": region, "year": year, "rainfall_mm": round(rainfall, 1),
                "temp_anomaly_c": round(temp_anomaly, 2), "gdp_per_capita": round(gdp, 1),
                "crop_yield_t_ha": round(yield_t, 3),
                "resilience_index": round(resilience, 4),
            })
    return pd.DataFrame(rows)


def main() -> Path:
    out = Path("data/raw/food_system.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"wrote food system data -> {out}")
    return out


if __name__ == "__main__":
    main()
