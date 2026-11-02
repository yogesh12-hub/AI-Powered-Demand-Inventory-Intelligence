from __future__ import annotations

import io
import zipfile

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_zip_bytes(tables: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in tables.items():
            if df is not None and len(df):
                z.writestr(f"{name}.csv", to_csv_bytes(df))
    return buf.getvalue()
