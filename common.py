"""Shared helpers: column names, data loading and cleaning."""
import numpy as np
import pandas as pd

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
COLUMNS = FEATURES + ["target"]

# Plain-English names, used on the web pages
LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest pain type",
    "trestbps": "Resting blood pressure", "chol": "Cholesterol",
    "fbs": "Fasting blood sugar > 120", "restecg": "Resting ECG",
    "thalach": "Max heart rate", "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)", "slope": "ST slope",
    "ca": "Major vessels (0-3)", "thal": "Thalassemia",
}

DATASETS = {
    "cleveland": "Cleveland (held-out test)",
    "hungarian": "Hungarian",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}


def load_dataset(name):
    """Read one UCI file, mark missing values, make the target binary (0/1)."""
    df = pd.read_csv(f"data/processed.{name}.data", header=None,
                     names=COLUMNS, na_values=["?", "-9", "-9.0"])
    # In some files a value of 0 for these fields means "not measured"
    for col in ("trestbps", "chol"):
        df.loc[df[col] == 0, col] = np.nan
    df = df.apply(pd.to_numeric, errors="coerce")
    df["target"] = (df["target"] > 0).astype(int)   # severity 1-4 -> disease
    return df
