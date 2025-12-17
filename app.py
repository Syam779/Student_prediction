from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, json, os

# =========================
# APP SETUP
# =========================
app = Flask(__name__)

# =========================
# LOAD DATA ONCE
# =========================
DATA_PATH = os.path.join("data", "cleaned_dataset.csv")
MODEL_DIR = "model"

df = pd.read_csv(DATA_PATH)

# =========================
# LOAD MODEL & FEATURES
# =========================
model = joblib.load(os.path.join(MODEL_DIR, "high_performance_model.pkl"))

with open(os.path.join(MODEL_DIR, "model_features.json")) as f:
    FEATURES = json.load(f)

# =========================
# HELPER: ESTIMATE EXAM SCORE
# =========================
def estimate_exam_score(behaviour_score):
    low = df["FinalGrade"].quantile(0.10)
    high = df["FinalGrade"].quantile(0.90)
    est = low + (behaviour_score / 100) * (high - low)
    return round(float(est), 1)

# =========================
# HOME
# =========================
@app.route("/")
def home():
    kpi = {
        "total": len(df),
        "avg_grade": round(df["FinalGrade"].mean(), 2),
        "avg_attendance": round(df["Attendance"].mean(), 1),
        "high_performers": int((df["FinalGrade"] >= 75).sum())
    }
    return render_template("home.html", kpi=kpi)

# =========================
# PREDICTION + WHAT-IF
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = probability = score = risk_level = None
    advice = []
    breakdown = {}
    benchmark = {}
    form_values = {}

    current_exam_score = improved_exam_score = improved_score = None
    avg = df.mean(numeric_only=True)

    if request.method == "POST":
        values, data = {}, []

        for f in FEATURES:
            val = float(request.form.get(f, 0))
            values[f] = val
            form_values[f] = val
            data.append(val)

        X = np.array(data).reshape(1, -1)
        pred = model.predict(X)[0]

        prob = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else 0

        prediction = "High Performer" if pred == 1 else "Not High Performer"
        probability = round(prob * 100, 2)

        breakdown = {
            "Study Hours": round((values["StudyHours"] / 20) * 25, 1),
            "Attendance": round((values["Attendance"] / 100) * 25, 1),
            "Assignments": round((values["AssignmentCompletion"] / 100) * 25, 1),
            "Stress Management": round(((10 - values["StressLevel"]) / 10) * 25, 1),
        }

        score = round(sum(breakdown.values()), 1)

        risk_level = (
            "Excellent" if score >= 80 else
            "Good" if score >= 65 else
            "Average" if score >= 50 else
            "At Risk"
        )

        if values["StudyHours"] < avg["StudyHours"]:
            advice.append("Increase weekly study hours.")
        if values["Attendance"] < 80:
            advice.append("Improve class attendance.")
        if values["AssignmentCompletion"] < 70:
            advice.append("Submit assignments more consistently.")
        if values["StressLevel"] > 7:
            advice.append("Reduce stress through time management.")

        improved_values = values.copy()
        if "Increase weekly study hours." in advice:
            improved_values["StudyHours"] = min(values["StudyHours"] + 5, 40)
        if "Improve class attendance." in advice:
            improved_values["Attendance"] = max(values["Attendance"], 90)
        if "Submit assignments more consistently." in advice:
            improved_values["AssignmentCompletion"] = max(values["AssignmentCompletion"], 85)
        if "Reduce stress through time management." in advice:
            improved_values["StressLevel"] = max(values["StressLevel"] - 2, 1)

        improved_score = round(sum([
            (improved_values["StudyHours"] / 20) * 25,
            (improved_values["Attendance"] / 100) * 25,
            (improved_values["AssignmentCompletion"] / 100) * 25,
            ((10 - improved_values["StressLevel"]) / 10) * 25
        ]), 1)

        current_exam_score = estimate_exam_score(score)
        improved_exam_score = estimate_exam_score(improved_score)

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability,
        score=score,
        risk_level=risk_level,
        breakdown=breakdown,
        advice=advice,
        form_values=form_values,
        current_exam_score=current_exam_score,
        improved_exam_score=improved_exam_score,
        improved_score=improved_score
    )

# =========================
# DEEP INSIGHTS
# =========================
@app.route("/insights")
def insights():
    feature_importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    fig = px.bar(
        feature_importance,
        title="Feature Importance",
        labels={"value": "Importance", "index": "Feature"}
    )

    chart = fig.to_html(full_html=False)

    return render_template("insights.html", chart=chart)

# =========================
# TRENDS
# =========================
@app.route("/trends")
def trends():
    fig = px.histogram(
        df,
        x="FinalGrade",
        nbins=20,
        title="Distribution of Final Grades"
    )

    chart = fig.to_html(full_html=False)
    return render_template("trends.html", chart=chart)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
