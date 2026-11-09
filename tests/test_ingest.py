"""Builds a miniature Favorita-shaped dataset (closed days, oil gaps, moved holidays, multi-word
state names) and runs ingest -> pipeline -> train -> forecast in an isolated store."""
import numpy as np
import pandas as pd
import pytest
from insightpilot import config
from insightpilot.data.ingest_favorita import ingest
from insightpilot.data.pipeline import run as run_pipeline
from insightpilot.ml import predict
from insightpilot.ml.train import train


@pytest.fixture()
def favorita_dir(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", "2016-12-31")
    dates = dates[~((dates.month == 12) & (dates.day == 25))]          # stores closed on Dec 25
    stores = pd.DataFrame({"store_nbr": [1, 2, 3, 4], "city": list("QQGG"),
                           "state": ["Pichincha", "Pichincha", "Santo Domingo de los Tsachilas", "Guayas"],
                           "type": ["A", "B", "A", "C"], "cluster": [1, 2, 3, 4]})
    fam = ["BEVERAGES", "LIQUOR,WINE,BEER", "GROCERY I", "GROCERY II", "BOOKS"]
    idx = pd.MultiIndex.from_product([dates, stores.store_nbr, fam], names=["date", "store_nbr", "family"])
    tr = idx.to_frame(index=False)
    tr["sales"] = rng.poisson(20 * (1 + 0.3 * (tr["date"].dt.dayofweek >= 5)) + tr["store_nbr"] * 3) + rng.random(len(tr))
    tr["onpromotion"] = rng.integers(0, 5, len(tr))
    tr.insert(0, "id", range(len(tr)))
    tr.to_csv(tmp_path / "train.csv", index=False)
    stores.to_csv(tmp_path / "stores.csv", index=False)
    oil = pd.DataFrame({"date": pd.bdate_range("2015-01-01", "2016-12-31")})   # business days only
    oil["dcoilwtico"] = 50 + rng.normal(0, 1, len(oil))
    oil.loc[0, "dcoilwtico"] = np.nan
    oil.to_csv(tmp_path / "oil.csv", index=False)
    pd.DataFrame({"date": ["2015-01-01", "2015-05-01", "2015-06-10", "2016-01-01"],
                  "type": ["Holiday", "Holiday", "Holiday", "Work Day"],
                  "locale": ["National", "National", "Local", "National"],
                  "locale_name": ["Ecuador"] * 4, "description": ["a", "b", "c", "d"],
                  "transferred": [False, True, False, False]}).to_csv(tmp_path / "holidays_events.csv", index=False)
    return tmp_path


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("INSIGHTPILOT_HOME", str(tmp_path / "store"))
    predict.reset_cache()
    yield
    predict.reset_cache()


def test_ingest_produces_canonical_schema(favorita_dir, tmp_path):
    out = ingest(favorita_dir, out=tmp_path / "raw.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["sale_date", "store_id", "region", "channel", "sku", "product_name",
                                "category", "units", "promo_items", "is_holiday", "oil_price"]
    assert df["oil_price"].notna().all()                                   # gaps forward/back filled
    assert not df["sale_date"].str.endswith("12-25").any()                 # closed days stay missing
    h = df.drop_duplicates("sale_date").set_index("sale_date")["is_holiday"]
    assert h["2015-01-01"] == 1 and h["2015-05-01"] == 0                   # moved holiday not counted
    assert h["2016-01-01"] == 0                                            # Work Day is not a holiday
    assert h["2015-06-10"] == 0                                            # local holiday ignored
    assert set(df["category"]) == {"Beverages", "Grocery", "General Merchandise"}


def test_ingest_validates_columns(favorita_dir, tmp_path):
    (favorita_dir / "stores.csv").write_text("store_nbr,city\n1,Quito\n")
    with pytest.raises(ValueError, match="missing columns"):
        ingest(favorita_dir, out=tmp_path / "raw.csv")


def test_real_data_end_to_end(favorita_dir, isolated_store):
    ingest(favorita_dir)                       # writes into the isolated store's raw path
    counts = run_pipeline()
    assert counts["silver_sales"] == counts["bronze_sales"]               # clean source drops nothing
    metrics = train()
    assert metrics["val_wmape"] < metrics["baseline_wmape_roll28"]
    r = predict.forecast("grocery i", "santo domingo de los tsachilas", 7)
    assert r["region"] == "Santo Domingo de los Tsachilas" and len(r["daily"]) == 7
    assert r["history_ends"] == "2016-12-31"
