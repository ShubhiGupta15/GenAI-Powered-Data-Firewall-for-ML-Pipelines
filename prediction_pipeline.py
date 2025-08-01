# prediction_pipeline.py
import pandas as pd
import joblib
import json
import os
import numpy as np

ARTIFACTS_DIR = "model_artifacts"

model = joblib.load(os.path.join(ARTIFACTS_DIR, "final_ml_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "standard_scaler.pkl"))
encoders = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "model_features.json")) as f:
    expected_features = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, "column_dtypes.json")) as f:
    expected_dtypes = json.load(f)

with open(os.path.join(ARTIFACTS_DIR, "classification_threshold.txt")) as f:
    threshold = float(f.read().strip())


def add_derived_features(df):
    df = df.copy()
    if 'order_date_(DateOrders)' in df.columns:
        df['order_date_(DateOrders)'] = pd.to_datetime(df['order_date_(DateOrders)'], errors='coerce')
        df['is_holiday_week'] = df['order_date_(DateOrders)'].dt.isocalendar().week.isin([1, 52]).astype(int)
        df['order_day'] = df['order_date_(DateOrders)'].dt.weekday.astype('Int64')
    else:
        df['is_holiday_week'] = 0
        df['order_day'] = 0

    if 'shipping_date_(DateOrders)' in df.columns:
        df['shipping_date_(DateOrders)'] = pd.to_datetime(df['shipping_date_(DateOrders)'], errors='coerce')
        df['ship_month'] = df['shipping_date_(DateOrders)'].dt.month.astype('Int64')
    else:
        df['ship_month'] = 0

    return df


def clean_columns(df):
    derived = ['is_holiday_week', 'order_day', 'ship_month']
    keep = set(expected_features + derived)
    return df[[col for col in df.columns if col in keep]]


def preprocess_input(df):
    df = add_derived_features(df)
    df = clean_columns(df)

    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val if not np.isnan(mean_val) else 0)

    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col].astype(str))
            except ValueError as e:
                unseen = set(df[col].dropna().unique()) - set(encoder.classes_)
                raise ValueError(f"Unseen labels in '{col}': {unseen}") from e

    for col in expected_features:
        if col in df.columns:
            expected_dtype = expected_dtypes[col]
            try:
                df[col] = df[col].astype(expected_dtype)
            except Exception as e:
                raise ValueError(f"Column '{col}' cannot be converted to {expected_dtype}: {e}")

    numeric_cols = [col for col in scaler.feature_names_in_ if col in df.columns]
    if numeric_cols:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df[expected_features]


def predict_pipeline(df):
    processed = preprocess_input(df)
    probs = model.predict_proba(processed)[:, 1]
    preds = (probs >= threshold).astype(int)
    return preds, probs
