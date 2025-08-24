import pandas as pd
from pathlib import Path
import numpy as np

DATA_DIR = Path("Data")   # <-- use capital "D" to match your folder name
RAW = DATA_DIR / "supply_chain_data.csv"
OUT = DATA_DIR / "prepared_data.csv"
RANDOM_SEED = 42

def main():
    df = pd.read_csv(RAW)

    # Standardize columns (strip spaces, lower, underscores)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename to match PDF terminology where possible
    df = df.rename(columns={
        "sku": "productid",
        "number_of_products_sold": "historicalsales"
    })

    # Create the extra columns if missing
    if "date" not in df.columns:
        rng = pd.date_range(start="2023-01-01", periods=len(df), freq="MS")
        df["date"] = rng

    if "promotion" not in df.columns:
        df["promotion"] = 0

    if "weather" not in df.columns:
        df["weather"] = "NA"

    if "economicindicators" not in df.columns:
        df["economicindicators"] = "Stable"

    # Basic cleaning: drop duplicates
    df = df.drop_duplicates(subset=["productid"])

    # Ensure proper type for date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Guard: keep only rows with valid target
    df = df[pd.notnull(df["historicalsales"])]

    # --- ✅ ADD DATE-DERIVED FEATURES ---
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter

    # Save prepared data with new columns
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved prepared data -> {OUT.resolve()}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()