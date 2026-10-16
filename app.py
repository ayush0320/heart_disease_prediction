"""Step 2: small Flask app. Run train.py first, then:  python app.py"""
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from common import FEATURES, LABELS

app = Flask(__name__)
bundle = joblib.load("artifacts/model.pkl")
model, imputer, explainer = bundle["model"], bundle["imputer"], bundle["explainer"]
results = json.load(open("artifacts/results.json"))

# Dropdown options for the form: field -> list of (value, text)
CHOICES = {
    "sex": [(1, "Male"), (0, "Female")],
    "cp": [(1, "1 - Typical angina"), (2, "2 - Atypical angina"),
           (3, "3 - Non-anginal pain"), (4, "4 - Asymptomatic")],
    "fbs": [(0, "No"), (1, "Yes")],
    "restecg": [(0, "0 - Normal"), (1, "1 - ST-T abnormality"), (2, "2 - LV hypertrophy")],
    "exang": [(0, "No"), (1, "Yes")],
    "slope": [(1, "1 - Upsloping"), (2, "2 - Flat"), (3, "3 - Downsloping")],
    "ca": [(0, "0"), (1, "1"), (2, "2"), (3, "3")],
    "thal": [(3, "3 - Normal"), (6, "6 - Fixed defect"), (7, "7 - Reversible defect")],
}
NUMERIC = {"age": (29, 77), "trestbps": (90, 200), "chol": (120, 570),
           "thalach": (70, 205), "oldpeak": (0, 6.5)}

EXAMPLES = {   # two sample patients taken from the Cleveland file
    "Example: lower-risk patient": dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1,
        restecg=2, thalach=150, exang=0, oldpeak=2.3, slope=3, ca=0, thal=6),
    "Example: higher-risk patient": dict(age=67, sex=1, cp=4, trestbps=160, chol=286, fbs=0,
        restecg=2, thalach=108, exang=1, oldpeak=1.5, slope=2, ca=3, thal=3),
}


@app.route("/")
def home():
    return render_template("index.html", choices=CHOICES, numeric=NUMERIC,
                           labels=LABELS, examples=EXAMPLES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {f: float(request.form[f]) for f in FEATURES}
    except (KeyError, ValueError):
        return render_template("index.html", choices=CHOICES, numeric=NUMERIC, labels=LABELS,
                               examples=EXAMPLES, error="Please fill in every field with a valid number."), 400

    row = imputer.transform(pd.DataFrame([values], columns=FEATURES))
    prob = float(model.predict_proba(row)[0][1])

    # ---- SHAP: how much did each input push this patient's risk up or down? ----
    sv = np.array(explainer.shap_values(row))
    sv = sv[..., 1] if sv.ndim == 3 else sv          # values for the "disease" class
    contrib = sv[0]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    rows = sorted(({"label": LABELS[f], "value": values[f], "shap": float(c) * 100}
                   for f, c in zip(FEATURES, contrib)), key=lambda r: -abs(r["shap"]))
    scale = max(abs(r["shap"]) for r in rows) or 1
    for r in rows:
        r["width"] = round(abs(r["shap"]) / scale * 100)
    return render_template("result.html", prob=round(prob * 100, 1), flagged=prob >= 0.5,
                           base=round(base * 100, 1), rows=rows, model_name=bundle["model_name"])


@app.route("/validation")
def validation():
    imp = results["global_importance"]
    top = max(i["importance"] for i in imp)
    for i in imp:
        i["width"] = round(i["importance"] / top * 100)
    return render_template("validation.html", r=results, importance=imp)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
