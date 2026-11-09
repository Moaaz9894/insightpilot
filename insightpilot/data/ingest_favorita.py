"""Convert the Kaggle 'Store Sales - Time Series Forecasting' (Favorita) files into the project's
canonical raw CSV.  Usage (PowerShell):

    $env:INSIGHTPILOT_HOME = "$PWD\\store\\favorita"
    python -m insightpilot.data.ingest_favorita --src D:\\path\\to\\favorita [--since 2016-01-01]
    python -m insightpilot.bootstrap

Mapping: state -> region, store type -> channel, product family -> sku/product_name,
sales -> units, onpromotion -> promo_items, national holidays -> is_holiday, oil price (forward-filled).
NOTE: the dataset has no prices, so the project works with units (not revenue)."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ..config import raw_csv

CATEGORY_MAP = {
    "BEVERAGES": "Beverages", "LIQUOR,WINE,BEER": "Beverages",
    "BREAD/BAKERY": "Fresh Food", "DAIRY": "Fresh Food", "DELI": "Fresh Food", "EGGS": "Fresh Food",
    "MEATS": "Fresh Food", "POULTRY": "Fresh Food", "PREPARED FOODS": "Fresh Food",
    "PRODUCE": "Fresh Food", "SEAFOOD": "Fresh Food", "FROZEN FOODS": "Fresh Food",
    "GROCERY I": "Grocery", "GROCERY II": "Grocery", "CELEBRATION": "Grocery",
    "CLEANING": "Home & Care", "HOMECARE": "Home & Care", "HOME AND KITCHEN I": "Home & Care",
    "HOME AND KITCHEN II": "Home & Care", "HOME APPLIANCES": "Home & Care", "HARDWARE": "Home & Care",
    "LAWN AND GARDEN": "Home & Care", "PERSONAL CARE": "Home & Care", "BEAUTY": "Home & Care",
    "BABY CARE": "Home & Care", "PET SUPPLIES": "Home & Care",
}
REQUIRED = {
    "train.csv": ["date", "store_nbr", "family", "sales", "onpromotion"],
    "stores.csv": ["store_nbr", "state", "type"],
    "oil.csv": ["date", "dcoilwtico"],
    "holidays_events.csv": ["date", "type", "locale", "transferred"],
}


def _read(src: Path, name: str, **kw) -> pd.DataFrame:
    path = src / name
    if not path.exists():
        raise FileNotFoundError(f"{name} not found in {src}")
    df = pd.read_csv(path, **kw)
    missing = set(REQUIRED[name]) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns {sorted(missing)}; found {list(df.columns)}")
    return df


def ingest(src, out=None, since: str | None = None) -> Path:
    src = Path(src)
    train = _read(src, "train.csv", usecols=lambda c: c in REQUIRED["train.csv"], parse_dates=["date"])
    if since:
        train = train[train["date"] >= pd.Timestamp(since)]
    stores = _read(src, "stores.csv")
    oil = _read(src, "oil.csv", parse_dates=["date"]).set_index("date")["dcoilwtico"]
    hol = _read(src, "holidays_events.csv", parse_dates=["date"])

    df = train.merge(stores[["store_nbr", "state", "type"]], on="store_nbr", how="left")
    if df["state"].isna().any():
        raise ValueError("Some store_nbr values in train.csv are missing from stores.csv")

    # oil is only published on business days: forward-fill (then back-fill the leading gap)
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    oil = oil[~oil.index.duplicated()].reindex(full).ffill().bfill()
    # national holidays that were actually celebrated on that date (not moved) and not make-up work days
    transferred = hol["transferred"].astype(str).str.lower().eq("true")
    nat = hol[(hol["locale"] == "National") & (~transferred) & (hol["type"] != "Work Day")]
    holiday_dates = set(nat["date"])

    family = df["family"].astype(str)
    out_df = pd.DataFrame({
        "sale_date": df["date"].dt.strftime("%Y-%m-%d"),
        "store_id": "STR-" + df["store_nbr"].astype(str).str.zfill(2),
        "region": df["state"].astype(str),
        "channel": "Type " + df["type"].astype(str),
        "sku": family,
        "product_name": family.str.title(),
        "category": family.map(lambda f: CATEGORY_MAP.get(f, "General Merchandise")),
        "units": df["sales"].round().astype("int64"),
        "promo_items": df["onpromotion"].fillna(0).astype("int64"),
        "is_holiday": df["date"].isin(holiday_dates).astype(int),
        "oil_price": df["date"].map(oil).round(2),
    })
    path = Path(out) if out else raw_csv()
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)
    print(f"wrote {len(out_df):,} rows -> {path}\n"
          f"  dates {out_df.sale_date.min()} .. {out_df.sale_date.max()} | stores {out_df.store_id.nunique()} "
          f"| regions {out_df.region.nunique()} | families {out_df.sku.nunique()} "
          f"| holiday-days {int(out_df.drop_duplicates('sale_date').is_holiday.sum())}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing train.csv, stores.csv, oil.csv, holidays_events.csv")
    ap.add_argument("--since", help="optional start date (YYYY-MM-DD) to shrink the data for faster dev runs")
    a = ap.parse_args()
    ingest(a.src, since=a.since)
