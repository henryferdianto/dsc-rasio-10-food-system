from src.features import build, climate_anomaly, rolling_yield
from src.sample_data import make
from src.train import train


def test_features_add_columns():
    df = make()
    out = climate_anomaly(df, "rainfall_mm")
    assert "rainfall_mm_anomaly" in out.columns
    assert f"crop_yield_t_ha_roll3" in rolling_yield(df, "crop_yield_t_ha").columns


def test_build_one_hot_encodes_region():
    out = build(make())
    assert any(c.startswith("region_") for c in out.columns)


def test_model_beats_mean_baseline():
    _, _, cv_rmse, _ = train(make())
    assert cv_rmse < 0.3
