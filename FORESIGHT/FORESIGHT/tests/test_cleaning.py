import numpy as np
import pandas as pd

from src.config import Settings
from src.data.cleaning import clean_dataset
from src.data.preprocessing import build_weekly, enrich_sales
import pytest


def sales(rows):
    return pd.DataFrame(rows, columns=["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"])


def test_cleaning_logs_every_drop_and_never_zero_fills():
    df = sales([("2024-01-01", " a ", "5", "50", "10", "yes"), ("2024-01-02", "A", None, "1", "1", "0"),
                ("2024-01-03", "a", "-3", "1", "1", "0"), ("bad", "a", "1", "1", "1", "0"), ("2024-01-04", "b", "2", "", "", "1")])
    out, log = clean_dataset(df, "sales")
    assert len(out) == 2                                  # only the two valid rows remain
    assert set(out.sku_id) == {"A", "B"}                  # normalised
    assert out.loc[out.sku_id == "B", "revenue"].isna().all()   # blank revenue stays blank (not 0)
    steps = {l["step"] for l in log}
    assert {"missing_units_sold", "negative_units_sold", "date_date"} <= steps


def test_key_duplicates_aggregated_and_exact_dropped():
    df = sales([("2024-01-01", "a", "5", "50", "10", "0"), ("2024-01-01", "a", "5", "50", "10", "0"), ("2024-01-01", "a", "3", "30", "10", "1")])
    out, log = clean_dataset(df, "sales")
    assert len(out) == 1 and out.units_sold.iloc[0] == 8 and bool(out.promo_flag.iloc[0])
    assert {"exact_duplicates", "key_duplicates"} <= {l["step"] for l in log}


def test_weekly_excludes_partial_weeks_and_gap_policy():
    days = pd.date_range("2024-01-03", "2024-02-20")      # starts Wed, ends Tue -> partial edge weeks
    rows = [(d.strftime("%Y-%m-%d"), "a", "1", "1", "1", "0") for d in days if not (pd.Timestamp("2024-01-22") <= d <= pd.Timestamp("2024-01-28"))]
    clean, _ = clean_dataset(sales(rows), "sales")
    w_zero, _ = build_weekly(clean, Settings(missing_weeks_as_zero=True))
    assert w_zero.week_start.min() == pd.Timestamp("2024-01-08") and w_zero.week_start.max() == pd.Timestamp("2024-02-12")
    assert (w_zero.set_index("week_start").loc["2024-01-22", "units"] == 0)
    w_nan, log = build_weekly(clean, Settings(missing_weeks_as_zero=False))
    assert np.isnan(w_nan.set_index("week_start").loc["2024-01-22", "units"])


def test_enrich_requires_unique_master_and_keeps_rows():
    s = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 2), "sku_id": ["A", "B"], "units_sold": [1.0, 2.0]})
    m = pd.DataFrame({"sku_id": ["A", "B"], "category": ["x", "y"]})
    assert len(enrich_sales(s, m)) == 2
    with pytest.raises(AssertionError):
        enrich_sales(s, pd.DataFrame({"sku_id": ["A", "A"], "category": ["x", "y"]}))


def test_empty_after_cleaning_yields_zero_rows():
    out, _ = clean_dataset(sales([("bad", "a", "1", "1", "1", "0")]), "sales")
    assert out.empty
