# 03_evaluate.py (replace your current main() with this)
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error

DATA = Path("data/prepared_data.csv")
ART = Path("artifacts")

def _create_date_features_from_date(df, cols):
    """Try to create missing date-derived features from df['date'].
    This creates columns with the exact names requested in `cols` when possible.
    Returns list of columns actually created.
    """
    created = []
    if "date" not in df.columns:
        return created

    # ensure datetime dtype
    df['date'] = pd.to_datetime(df['date'])

    for c in cols:
        if c in df.columns:
            continue
        # common mappings - create with the requested name
        if c == "month":
            df[c] = df['date'].dt.month
            created.append(c)
        elif c in ("dayofweek", "weekday", "day_of_week"):
            df[c] = df['date'].dt.dayofweek
            created.append(c)
        elif c == "quarter":
            df[c] = df['date'].dt.quarter
            created.append(c)
        elif c in ("day", "day_of_month"):
            df[c] = df['date'].dt.day
            created.append(c)
        elif c == "year":
            df[c] = df['date'].dt.year
            created.append(c)
        elif "week" in c:  # covers weekofyear, week_number, etc.
            # pandas returns isocalendar DataFrame in newer versions
            try:
                df[c] = df['date'].dt.isocalendar().week.astype(int)
            except AttributeError:
                df[c] = df['date'].dt.week.astype(int)  # older pandas
            created.append(c)
        # add other mappings here if you know them
    return created

def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    schema = json.loads((ART / "feature_schema.json").read_text())

    # quick debug prints so you can see what's happening
    print("Feature schema keys:", list(schema.keys()))
    print("Requested feature_cols:", schema.get("feature_cols"))
    print("Requested target_col:", schema.get("target_col"))
    print("Columns in prepared_data.csv:", list(df.columns))

    # try to create any missing date-derived columns (month, dayofweek, quarter, ...)
    missing = [c for c in schema["feature_cols"] if c not in df.columns]
    if missing:
        print("Missing feature columns before attempting creation:", missing)
        created = _create_date_features_from_date(df, missing)
        if created:
            print("Created date-derived columns:", created)

    # recompute missing after attempted creation
    missing = [c for c in schema["feature_cols"] if c not in df.columns]
    if missing:
        # give a clear, actionable error
        raise KeyError(
            f"Missing feature columns required by feature_schema.json: {missing}\n"
            "Either re-run your data preparation step that generates these features, "
            "or update feature_schema.json to match the actual columns in prepared_data.csv.\n"
            "To inspect the schema file run: print((ART / 'feature_schema.json').read_text())"
        )

    preprocessor = joblib.load(ART / "preprocessor.joblib")
    model = keras.models.load_model(ART / "demand_forecasting_model.keras")

    X = df[schema["feature_cols"]]
    y = df[schema["target_col"]].astype(float)

    # Simple train/test split to match training
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # transform and predict
    X_test_proc = preprocessor.transform(X_test)
    preds = model.predict(X_test_proc).ravel()

    mse = mean_squared_error(y_test, preds)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")

    # Plot true vs predicted
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.7)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Sales")
    plt.grid(True)
    plt.tight_layout()
    (ART / "true_vs_pred.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ART / "true_vs_pred.png")
    print(f"Saved plot -> {str((ART / 'true_vs_pred.png').resolve())}")

if __name__ == "__main__":
    main()