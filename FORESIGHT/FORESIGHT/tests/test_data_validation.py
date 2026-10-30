import pandas as pd

from src.data.column_mapping import apply_mapping, auto_map, mapping_problems
from src.data.schema_validation import parse_dates, validate


def codes(rep):
    return {i.code for i in rep.issues}


def test_missing_required_columns_is_error():
    rep = validate(pd.DataFrame({"date": ["2024-01-01"], "sku_id": ["a"]}), "sales")
    assert not rep.ok and "missing_columns" in codes(rep)


def test_empty_dataset_is_error():
    rep = validate(pd.DataFrame({"date": [], "sku_id": [], "units_sold": []}), "sales")
    assert not rep.ok and "empty" in codes(rep)


def test_invalid_numeric_types_flagged():
    df = pd.DataFrame({"date": ["2024-01-01"] * 3, "sku_id": list("abc"), "units_sold": ["x", "y", "z"]})
    rep = validate(df, "sales")
    assert not rep.ok and "non_numeric" in codes(rep)


def test_partial_bad_numbers_warn_and_negative_flagged():
    df = pd.DataFrame({"date": ["2024-01-01"] * 3, "sku_id": list("abc"), "units_sold": ["5", "oops", "-2"]})
    rep = validate(df, "sales")
    assert rep.ok and {"non_numeric", "negative"} <= codes(rep)


def test_date_parsing_dayfirst_vs_monthfirst():
    s = pd.Series(["03/04/2024", "25/12/2024"])
    assert parse_dates(s, dayfirst=True).iloc[0] == pd.Timestamp("2024-04-03")
    assert parse_dates(s, dayfirst=False).iloc[0] == pd.Timestamp("2024-03-04")
    assert parse_dates(pd.Series(["not a date"])).isna().all()


def test_unparseable_dates_error_when_all_bad():
    df = pd.DataFrame({"date": ["zzz", "yyy"], "sku_id": ["a", "b"], "units_sold": ["1", "2"]})
    assert "date_parse" in codes(validate(df, "sales")) and not validate(df, "sales").ok


def test_duplicates_detected():
    df = pd.DataFrame({"date": ["2024-01-01"] * 3, "sku_id": ["a", "a", "A "], "units_sold": ["1", "1", "4"]})
    c = codes(validate(df, "sales"))
    assert "exact_duplicates" in c and "key_duplicates" in c and "sku_variants" in c


def test_blank_values_not_converted_to_zero():
    df = pd.DataFrame({"date": ["2024-01-01"] * 2, "sku_id": ["a", "b"], "units_sold": ["3", None]})
    rep = validate(df, "sales")
    assert "required_missing" in codes(rep)


def test_column_mapping_aliases_and_apply():
    m = auto_map(["Order Date", "Item", "Qty", "Amount", "Price"], "sales")
    assert m["date"] == "Order Date" and m["sku_id"] == "Item" and m["units_sold"] == "Qty" and m["revenue"] == "Amount"
    df = pd.DataFrame({"Order Date": ["2024-01-01"], "Item": ["a"], "Qty": ["1"], "Amount": ["9"], "Price": ["9"]})
    out = apply_mapping(df, m)
    assert {"date", "sku_id", "units_sold"} <= set(out.columns)


def test_mapping_problems_required_and_duplicates():
    assert mapping_problems({"date": None, "sku_id": "x", "units_sold": "x"}, "sales")
