"""Feature engineering shared by training and serving (no train/serve skew).
lag/rolling features are shifted by 28 days, so any horizon <= 28 days can be
forecast in one shot from real history, with no recursive error build-up."""
from __future__ import annotations
import pandas as pd

LAG = 28
FEATURES = ["dow", "month", "doy", "promo_flag", "avg_price",
            "lag_28", "roll_28", "sku_code", "region_code"]


def build_features(df: pd.DataFrame, skus: list[str], regions: list[str]) -> pd.DataFrame:
    df = df.sort_values(["sku", "region", "sale_date"]).copy()
    g = df.groupby(["sku", "region"])["units"]
    df["lag_28"] = g.shift(LAG)
    df["roll_28"] = g.transform(lambda s: s.shift(LAG).rolling(28, min_periods=7).mean())
    d = pd.to_datetime(df["sale_date"])
    df["dow"], df["month"], df["doy"] = d.dt.dayofweek, d.dt.month, d.dt.dayofyear
    df["sku_code"] = df["sku"].map({s: i for i, s in enumerate(skus)})
    df["region_code"] = df["region"].map({r: i for i, r in enumerate(regions)})
    return df
