"""Step 1: train on Cleveland, compare models, explain (SHAP), validate on other hospitals."""
import json
import joblib
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import DATASETS, FEATURES, LABELS, load_dataset

SEED = 42


def metrics(y, pred, prob):
    out = {"accuracy": accuracy_score(y, pred),
           "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0)}
    out["roc_auc"] = roc_auc_score(y, prob) if len(set(y)) > 1 else None
    return {k: (round(float(v), 3) if v is not None else None) for k, v in out.items()}


# ---- 1. Load Cleveland and split -------------------------------------------
cle = load_dataset("cleveland")
X, y = cle[FEATURES], cle["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# The imputer is fitted ONLY on Cleveland training data and reused unchanged
# for every other hospital (no information leaks from the external data).
imputer = SimpleImputer(strategy="median").fit(X_train)
Xtr, Xte = imputer.transform(X_train), imputer.transform(X_test)

# ---- 2. Compare candidate models (5-fold CV on the training split) -----------
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             eval_metric="logloss", random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
comparison = {}
for name, m in candidates.items():
    auc = cross_val_score(m, Xtr, y_train, cv=cv, scoring="roc_auc")
    rec = cross_val_score(m, Xtr, y_train, cv=cv, scoring="recall")
    m.fit(Xtr, y_train)
    t = metrics(y_test, m.predict(Xte), m.predict_proba(Xte)[:, 1])
    comparison[name] = {"cv_auc_mean": round(auc.mean(), 3), "cv_auc_std": round(auc.std(), 3),
                        "cv_recall_mean": round(rec.mean(), 3), "test": t}
    print(f"{name:20s} CV AUC {auc.mean():.3f} +/- {auc.std():.3f} | CV recall {rec.mean():.3f} | test {t}")

# SHAP's TreeExplainer needs a tree model, so we deploy the best TREE model by CV AUC.
tree_names = ["Decision Tree", "Random Forest", "XGBoost"]
best = max(tree_names, key=lambda n: comparison[n]["cv_auc_mean"])
model = candidates[best]
print("\nDeployed (explainable) model:", best)

# ---- 3. SHAP explainability --------------------------------------------------
explainer = shap.TreeExplainer(model)


def class1_values(sv):
    """Return SHAP values for the 'disease' class as a 2-D array (handles all shap versions)."""
    sv = np.array(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


sv_test = class1_values(explainer.shap_values(Xte))
global_imp = np.abs(sv_test).mean(axis=0)
global_importance = sorted(
    [{"feature": f, "label": LABELS[f], "importance": round(float(v), 4)}
     for f, v in zip(FEATURES, global_imp)], key=lambda d: -d["importance"])

# ---- 4. Cross-dataset validation (the model is NOT retrained) ---------------
external = {}
for key, title in DATASETS.items():
    if key == "cleveland":          # use ONLY the held-out Cleveland test split
        Xe, ye = X_test, y_test
    else:
        df = load_dataset(key)
        Xe, ye = df[FEATURES], df["target"]
    prob = model.predict_proba(imputer.transform(Xe))[:, 1]
    pred = (prob >= 0.5).astype(int)
    row = metrics(ye, pred, prob)
    row.update({"name": title, "n": int(len(Xe)),
                "missing_pct": round(float(Xe.isna().mean().mean() * 100), 1),
                "positive_pct": round(float(ye.mean() * 100), 1)})
    external[key] = row
    print(f"{key:12s} {row}")

# ---- 5. Save everything the web app needs -------------------------------------
joblib.dump({"model": model, "imputer": imputer, "explainer": explainer,
             "model_name": best}, "artifacts/model.pkl")
with open("artifacts/results.json", "w") as f:
    json.dump({"model_name": best, "comparison": comparison, "external": external,
               "global_importance": global_importance,
               "split": {"train": int(len(X_train)), "test": int(len(X_test))}}, f, indent=2)
print("\nSaved artifacts/model.pkl and artifacts/results.json")
