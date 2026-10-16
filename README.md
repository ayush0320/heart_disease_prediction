# Heart Disease Prediction - Prototype

Small demo of the project plus its two literature-driven improvements:
1. SHAP explainability - every prediction shows which inputs pushed the risk up or down.
2. Cross-dataset validation - the model is trained on Cleveland only, then tested on the Hungarian,
   Switzerland and VA Long Beach UCI datasets without retraining.

## Run
    pip install -r requirements.txt
    python train.py      # trains, compares models, computes SHAP, runs cross-dataset validation (~1 min)
    python app.py        # open http://127.0.0.1:5000

## Files
- `common.py`  column names, loading and cleaning of the UCI files (`?` and impossible zeros -> missing)
- `train.py`   split -> impute -> compare 4 models -> SHAP -> cross-dataset test -> saves `artifacts/`
- `app.py`     Flask app: `/` form, `/predict` result with SHAP bars, `/validation` results page
- `data/`      the four UCI processed heart-disease files (UCI Machine Learning Repository, CC BY 4.0)

## 2-minute demo order
1. `/`  click "Example: higher-risk patient" -> Predict. Show the percentage and the red/green SHAP bars.
2. Go back, click "Example: lower-risk patient" -> compare how the bars change.
3. `/validation` -> model comparison, global feature importance, then the cross-dataset table.
