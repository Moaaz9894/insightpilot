"""Synthetic retail / FMCG sales generator (no real company data).
Produces a deliberately 'dirty' raw CSV so the Bronze->Silver cleaning is real work."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import raw_csv

REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Supermarket", "Convenience", "Wholesale"]
PRODUCTS = [
    ("SKU-001", "Cola 330ml", "Beverages", 0.80),
    ("SKU-002", "Orange Soda 330ml", "Beverages", 0.80),
    ("SKU-003", "Sparkling Water 500ml", "Beverages", 0.50),
    ("SKU-004", "Energy Drink 250ml", "Beverages", 1.50),
    ("SKU-005", "Apple Juice 1L", "Juice", 1.60),
    ("SKU-006", "Potato Chips 100g", "Snacks", 1.20),
    ("SKU-007", "Salted Peanuts 80g", "Snacks", 0.90),
    ("SKU-008", "Fresh Milk 1L", "Dairy", 1.10),
    ("SKU-009", "Yogurt Cup 150g", "Dairy", 0.60),
    ("SKU-010", "Croissant 60g", "Bakery", 0.70),
]


def generate(days: int = 730, n_stores: int = 20, seed: int = 42, out=None):
    """Canonical raw schema (shared with the real-data ingest):
    sale_date, store_id, region, channel, sku, product_name, category, units,
    promo_items, is_holiday, oil_price"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    stores = pd.DataFrame({
        "store_id": [f"STR-{i + 1:02d}" for i in range(n_stores)],
        "region": [REGIONS[i % len(REGIONS)] for i in range(n_stores)],
        "channel": [CHANNELS[i % len(CHANNELS)] for i in range(n_stores)],
        "store_scale": rng.uniform(0.6, 1.8, n_stores),
    })
    prods = pd.DataFrame(PRODUCTS, columns=["sku", "product_name", "category", "base_price"])
    prods["sku_scale"] = rng.uniform(8, 30, len(prods))

    df = pd.MultiIndex.from_product(
        [dates, stores.index, prods.index], names=["sale_date", "s", "p"]
    ).to_frame(index=False)
    df = df.merge(stores, left_on="s", right_index=True).merge(prods, left_on="p", right_index=True)

    dow = df["sale_date"].dt.dayofweek.to_numpy()
    doy = df["sale_date"].dt.dayofyear.to_numpy()
    hot = df["category"].isin(["Beverages", "Juice"]).to_numpy()
    season = np.where(hot, 1 + 0.30 * np.sin(2 * np.pi * (doy - 100) / 365),
                      1 + 0.08 * np.sin(2 * np.pi * (doy - 300) / 365))
    weekly = 1 + 0.18 * (dow >= 4)
    promo = rng.random(len(df)) < 0.10
    promo_items = np.where(promo, rng.integers(3, 16, len(df)), 0)
    md = df["sale_date"].dt.strftime("%m-%d")
    holiday = md.isin(["01-01", "05-01", "10-28", "12-24"]).to_numpy()
    oil = pd.Series(60 + np.cumsum(rng.normal(0, 0.8, days)).clip(-25, 25), index=dates).round(2)
    lam = (df["store_scale"].to_numpy() * df["sku_scale"].to_numpy() * weekly * season
           * np.where(promo, 1.4, 1.0) * np.where(holiday, 1.2, 1.0))
    df["units"] = rng.poisson(lam).astype(float)
    df["promo_items"] = promo_items
    df["is_holiday"] = holiday.astype(int)
    df["oil_price"] = df["sale_date"].map(oil)

    cols = ["sale_date", "store_id", "region", "channel", "sku", "product_name",
            "category", "units", "promo_items", "is_holiday", "oil_price"]
    df = df[cols].reset_index(drop=True)
    # --- inject realistic dirt for the Silver layer to fix ---
    dup = df.sample(frac=0.003, random_state=seed)
    df.loc[df.sample(frac=0.002, random_state=seed + 1).index, "units"] = np.nan
    df.loc[df.sample(frac=0.0005, random_state=seed + 2).index, "units"] = -1
    df = pd.concat([df, dup], ignore_index=True)
    df["sale_date"] = df["sale_date"].dt.strftime("%Y-%m-%d")

    path = out or raw_csv()
    df.to_csv(path, index=False)
    return path
