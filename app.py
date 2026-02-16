"""
Flask web app for life expectancy prediction (from task.ipynb).
Run train_model.py first (with data.csv), then: flask run
"""
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "pipeline.pkl")
COLS_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

pipeline = None
feature_columns = None


def load_model():
    global pipeline, feature_columns
    if os.path.isfile(PIPELINE_PATH) and os.path.isfile(COLS_PATH):
        pipeline = joblib.load(PIPELINE_PATH)
        feature_columns = joblib.load(COLS_PATH)
        return True
    return False


def _float(name, default=0.0):
    try:
        return float(request.form.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _int(name, default=0):
    try:
        return int(float(request.form.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def form_to_row():
    """Build one row dict from request form (same columns as training)."""
    return {
        "Country": request.form.get("Country", "Afghanistan").strip() or "Afghanistan",
        "Year": _int("Year", 2015),
        "Status": request.form.get("Status", "Developing") or "Developing",
        "Adult Mortality": _float("Adult Mortality", 200),
        "infant deaths": _float("infant deaths", 50),
        "Alcohol": _float("Alcohol", 2.0),
        "percentage expenditure": _float("percentage expenditure", 500),
        "Hepatitis B": _float("Hepatitis B", 80),
        "Measles": _float("Measles", 100),
        "BMI": _float("BMI", 20),
        "under-five deaths": _float("under-five deaths", 60),
        "Polio": _float("Polio", 80),
        "Total expenditure": _float("Total expenditure", 6),
        "Diphtheria": _float("Diphtheria", 80),
        "HIV/AIDS": _float("HIV/AIDS", 0.1),
        "GDP": _float("GDP", 2000),
        "Population": _float("Population", 1000000),
        "thinness  1-19 years": _float("thinness  1-19 years", 5),
        "thinness 5-9 years": _float("thinness 5-9 years", 5),
        "Income composition of resources": _float("Income composition of resources", 0.5),
        "Schooling": _float("Schooling", 10),
    }


@app.route("/")
def index():
    if not load_model():
        return render_template("index.html", model_loaded=False, stats=None)
    stats = None
    if os.path.isfile(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        target_col = "Life expectancy"
        if target_col not in df.columns:
            for c in df.columns:
                if "Life" in c and "expectancy" in c.lower():
                    df = df.rename(columns={c: target_col})
                    break
        if target_col in df.columns:
            total = len(df)
            mean_le = float(df[target_col].mean())
            by_status = df.groupby("Status", dropna=False)[target_col].agg(["count", "mean"]).to_dict("index")
            stats = {
                "total_rows": int(total),
                "mean_life_expectancy": round(mean_le, 2),
                "status_labels": list(by_status.keys()),
                "status_means": [round(float(by_status[k]["mean"]), 2) for k in by_status],
                "status_counts": [int(by_status[k]["count"]) for k in by_status],
            }
    return render_template("index.html", model_loaded=True, stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not load_model():
        return render_template("predict.html", model_loaded=False, result=None)
    if request.method == "GET":
        return render_template("predict.html", model_loaded=True, result=None)
    try:
        row = form_to_row()
        row_df = pd.DataFrame([row])[feature_columns]
        pred = pipeline.predict(row_df)[0]
        life_years = round(float(pred), 2)
        return render_template(
            "predict.html",
            model_loaded=True,
            result={"life_expectancy": life_years},
        )
    except Exception as e:
        return render_template(
            "predict.html", model_loaded=True, result=None, error=str(e)
        )


def _default_row():
    return {
        "Country": "Afghanistan", "Year": 2015, "Status": "Developing",
        "Adult Mortality": 200.0, "infant deaths": 50.0, "Alcohol": 2.0,
        "percentage expenditure": 500.0, "Hepatitis B": 80.0, "Measles": 100.0,
        "BMI": 20.0, "under-five deaths": 60.0, "Polio": 80.0,
        "Total expenditure": 6.0, "Diphtheria": 80.0, "HIV/AIDS": 0.1,
        "GDP": 2000.0, "Population": 1000000.0,
        "thinness  1-19 years": 5.0, "thinness 5-9 years": 5.0,
        "Income composition of resources": 0.5, "Schooling": 10.0,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not load_model():
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
    try:
        data = request.get_json() or request.form
        defaults = _default_row()
        row = {}
        for k in feature_columns:
            v = data.get(k, defaults.get(k, 0))
            if k in ("Country", "Status"):
                row[k] = str(v).strip() or defaults[k]
            elif k == "Year":
                row[k] = int(float(v))
            else:
                row[k] = float(v)
        row_df = pd.DataFrame([row])[feature_columns]
        pred = pipeline.predict(row_df)[0]
        return jsonify({"life_expectancy": round(float(pred), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5002)
