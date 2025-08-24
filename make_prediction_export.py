# make_predictions_export.py
import json
from pathlib import Path
import joblib
import pandas as pd
from tensorflow import keras

DATA = Path("Data") / "prepared_data.csv"
ART = Path("artifacts")
OUT = Path("predictions.csv")

def main():
    # Load preprocessing and model
    pre = joblib.load(ART / "preprocessor.joblib")
    schema = json.loads((ART / "feature_schema.json").read_text())
    model = keras.models.load_model(ART / "demand_forecasting_model.keras")

    # Load prepared dataset
    df = pd.read_csv(DATA, parse_dates=["date"])

    # Features + target
    X = df[schema["feature_cols"]].copy()
    y = df[schema["target_col"]].astype(float)

    # Transform and predict
    X_proc = pre.transform(X)
    preds = model.predict(X_proc).ravel()

    # Add predictions + error
    out = df.copy()
    out.rename(columns={"historicalsales":"Actual_Sales", "productid":"ProductID"}, inplace=True)
    out["Predicted_Sales"] = preds
    out["Error"] = out["Actual_Sales"] - out["Predicted_Sales"]

    out.to_csv(OUT, index=False)
    print(f"✅ Exported {OUT.resolve()} with {len(out)} rows")

if __name__ == "__main__":
    main()