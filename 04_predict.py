# 04_predict.py
"""
Usage example:

python 04_predict.py --from-row 0
or
python 04_predict.py --manual \
  --productid SKU12 --price 15.0 --availability 1 --stock_levels 35 --lead_times 3 \
  --order_quantities 40 --shipping_times 2 --shipping_costs 4.5 --lead_time 3 \
  --production_volumes 120 --manufacturing_lead_time 4 --manufacturing_costs 2.5 \
  --defect_rates 0.02 --costs 10.5 --date 2023-06-01 --promotion 0 \
  --weather NA --economicindicators Stable --product_type Makeup \
  --customer_demographics Women --shipping_carriers CarrierA \
  --supplier_name SupplierX --location Delhi --inspection_results Pass \
  --transportation_modes Road --routes Route1
"""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "Data" / "prepared_data.csv"
ART = BASE_DIR / "artifacts"

def load_artifacts():
    preprocessor = joblib.load(ART / "preprocessor.joblib")
    schema = json.loads((ART / "feature_schema.json").read_text())
    model = keras.models.load_model(ART / "demand_forecasting_model.keras")
    return preprocessor, schema, model

def df_from_row(df, row_idx, feature_cols):
    row = df.iloc[[row_idx]][feature_cols].copy()
    return row

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from-row", type=int, default=None, help="Use an existing row by index")
    p.add_argument("--manual", action="store_true", help="Provide manual feature values")
    p.add_argument("--date", type=str)
    # dynamic, we won’t enumerate every feature; manual mode builds a dict from args
    known = [
        "productid","price","availability","stock_levels","lead_times","order_quantities",
        "shipping_times","shipping_costs","lead_time","production_volumes",
        "manufacturing_lead_time","manufacturing_costs","defect_rates","costs",
        "promotion","weather","economicindicators","product_type","customer_demographics",
        "shipping_carriers","supplier_name","location","inspection_results",
        "transportation_modes","routes"
    ]
    for k in known:
        p.add_argument(f"--{k}", type=str)
    return p.parse_args()

def main():
    args = parse_args()
    preproc, schema, model = load_artifacts()
    df = pd.read_csv(DATA, parse_dates=["date"])

    if args.from_row is not None:
        X = df_from_row(df, args.from_row, schema["feature_cols"])
    elif args.manual:
        # Build one-row DataFrame from provided args; cast numerics when possible
        values = {}
        for col in schema["feature_cols"]:
            if col == "date":
                # not in feature_cols directly; we derived date features earlier
                continue
            v = getattr(args, col, None)
            if v is None:
                # fallback: use median/mode from data
                if col in df.columns:
                    if df[col].dtype.kind in "biufc":
                        v = float(df[col].median())
                    else:
                        v = df[col].mode().iloc[0]
                else:
                    v = 0
            # try to cast numerics
            try:
                if col not in ["productid","product_type","customer_demographics",
                               "shipping_carriers","supplier_name","location",
                               "inspection_results","transportation_modes",
                               "routes","weather","economicindicators","promotion"]:
                    v = float(v)
            except:
                pass
            values[col] = v
        X = pd.DataFrame([values])[schema["feature_cols"]]
    else:
        raise SystemExit("Provide --from-row or --manual.")

    # Rebuild date-derived features if needed (should already be in prepared file workflow)
    if "date" in df.columns and "month" in schema["date_cols"]:
        # make sure month/dayofweek/quarter exist if user constructed X manually
        if not set(schema["date_cols"]).issubset(X.columns):
            # fill from dataset median date
            base = pd.to_datetime(df["date"]).median()
            X = X.copy()
            X["month"] = base.month
            X["dayofweek"] = base.dayofweek
            X["quarter"] = base.quarter

    X_proc = preproc.transform(X)
    pred = model.predict(X_proc).ravel()[0]
    print(f"Predicted Sales: {pred:.4f}")

if __name__ == "__main__":
    main()