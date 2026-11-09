"""Feature engineering shared by training and serving (no train/serve skew).
lag/rolling features are shifted by 28 days, so any horizon <= 28 days can be
forecast in one shot from real history, with no recursive error build-up.
Series are pivoted onto a COMPLETE daily calendar first, so missing days (e.g. Dec 25 when
stores are closed) never shift the lags."""
from __future__ import annotations
import pandas as pd

LAG = 28
FEATURES = ["dow", "month", "doy", "dom", "is_payday", "promo_items", "is_holiday",
            "oil_price", "lag_28", "roll_28", "sku_code", "region_code"]


def _long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    out = wide.reset_index().melt(id_vars="sale_date", var_name="series", value_name=name)
    return out


def build_features(df: pd.DataFrame, skus: list[str], regions: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["series"] = df["sku"].astype(str) + "|" + df["region"].astype(str)

    wide = df.pivot(index="sale_date", columns="series", values="units")
    wide = wide.reindex(pd.date_range(df["sale_date"].min(), df["sale_date"].max(), freq="D"))
    wide.index.name = "sale_date"
    lag = wide.shift(LAG)
    roll = lag.rolling(28, min_periods=7).mean()
    df = (df.merge(_long(lag, "lag_28"), on=["sale_date", "series"], how="left")
            .merge(_long(roll, "roll_28"), on=["sale_date", "series"], how="left"))

    d = df["sale_date"]
    df["dow"], df["month"], df["doy"], df["dom"] = d.dt.dayofweek, d.dt.month, d.dt.dayofyear, d.dt.day
    df["is_payday"] = ((d.dt.day == 15) | d.dt.is_month_end).astype(int)  # public-sector pay days
    df["promo_items"] = df["promo_items"].fillna(0)
    df["is_holiday"] = df["is_holiday"].fillna(0)
    df["oil_price"] = df["oil_price"].fillna(df["oil_price"].median())
    df["sku_code"] = df["sku"].map({s: i for i, s in enumerate(skus)})
    df["region_code"] = df["region"].map({r: i for i, r in enumerate(regions)})
    return df.sort_values(["sku", "region", "sale_date"]).reset_index(drop=True)
