# 02_train.py
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA = Path("data/prepared_data.csv")
ART = Path("artifacts")
ART.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42
EPOCHS = 100
BATCH = 16
VAL_SPLIT = 0.2

def build_model(n_features: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)  # regression
    ])
    # Use mse loss only; no custom metrics to avoid deserialize issues
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    df = pd.read_csv(DATA, parse_dates=["date"])

    # Target and features as per PDF
    target_col = "historicalsales"

    # Use sensible features from your CSV; avoid leakage (exclude revenue_generated)
    # Numeric features present in your file
    numeric_cols = [
        "price", "availability", "stock_levels", "lead_times", "order_quantities",
        "shipping_times", "shipping_costs", "lead_time", "production_volumes",
        "manufacturing_lead_time", "manufacturing_costs", "defect_rates", "costs"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Date-derived features (month, weekday, quarter)
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    date_cols = ["month", "dayofweek", "quarter"]

    # Categorical features
    cat_cols = []
    for c in ["product_type","customer_demographics","shipping_carriers",
              "supplier_name","location","inspection_results",
              "transportation_modes","routes","promotion","weather","economicindicators"]:
        if c in df.columns:
            cat_cols.append(c)

    # Also include ProductID as categorical signal
    if "productid" in df.columns:
        cat_cols.append("productid")

    # Final feature set
    feature_cols = numeric_cols + date_cols + cat_cols

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)

    # Build preprocessing (OHE for categorical, scale numeric)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols + date_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    # Fit/transform
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Save the fitted preprocessor and schema for safe inference
    joblib.dump(preprocessor, ART / "preprocessor.joblib")
    schema = {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "date_cols": date_cols,
        "cat_cols": cat_cols,
        "target_col": target_col
    }
    (ART / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    # Build and train the model
    model = build_model(n_features=X_train_proc.shape[1])
    cb = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
    ]
    history = model.fit(
        X_train_proc, y_train,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=cb,
        verbose=1
    )

    # Save model in modern format
    model.save(ART / "demand_forecasting_model.keras")

    # Quick test MSE so we see training worked
    test_preds = model.predict(X_test_proc).ravel()
    mse = float(np.mean((y_test - test_preds) ** 2))
    print(f"Saved model & preprocessor to {ART.resolve()}")
    print(f"Holdout MSE: {mse:.4f}")

if __name__ == "__main__":
    main()