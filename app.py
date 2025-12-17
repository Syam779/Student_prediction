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
# HOME (ROOT PAGE)
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
# PREDICTION (ML)
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
# INSIGHTS
# =========================
@app.route("/insights", methods=["GET", "POST"])
def insights():
    df_i = df.copy()
    df_i["HighPerformer"] = df_i["FinalGrade"] >= 75

    numeric_cols = [
        "StudyHours",
        "Attendance",
        "AssignmentCompletion",
        "Motivation",
        "StressLevel",
        "FinalGrade"
    ]

    kpi = {
        "avg_grade": round(df_i["FinalGrade"].mean(), 2),
        "high_pct": round((df_i["FinalGrade"] >= 75).mean() * 100, 2),
        "avg_att": round(df_i["Attendance"].mean(), 2),
        "avg_study": round(df_i["StudyHours"].mean(), 2),
    }

    fig = heatmap = None
    submitted = False

    x_axis = "StudyHours"
    y_axis = "FinalGrade"
    chart_type = "Scatter"

    if request.method == "POST":
        submitted = True

        x_axis = request.form.get("x_axis")
        y_axis = request.form.get("y_axis")
        chart_type = request.form.get("chart_type")

        if chart_type == "Box":
            fig = px.box(df_i, x=x_axis, y=y_axis, color="HighPerformer")
        elif chart_type == "Histogram":
            fig = px.histogram(df_i, x=x_axis, color="HighPerformer")
        else:
            fig = px.scatter(df_i, x=x_axis, y=y_axis, color="HighPerformer")

        fig = fig.to_html(full_html=False)

        heatmap = px.imshow(df_i[numeric_cols].corr(), color_continuous_scale="RdBu")
        heatmap = heatmap.to_html(full_html=False)

    return render_template(
        "insights.html",
        kpi=kpi,
        fig=fig,
        heatmap=heatmap,
        columns=numeric_cols,
        x_axis=x_axis,
        y_axis=y_axis,
        chart_type=chart_type,
        submitted=submitted
    )

# =========================
# TRENDS
# =========================
@app.route("/trends", methods=["GET", "POST"])
def trends():
    df_t = df.copy()
    df_t["Period"] = (np.arange(len(df_t)) // 50) + 1
    df_t["HighPerformer"] = df_t["FinalGrade"] >= 75

    METRICS = {
        "FinalGrade": "Final Grade",
        "Attendance": "Attendance (%)",
        "StudyHours": "Study Hours",
        "AssignmentCompletion": "Assignment Completion (%)",
        "Motivation": "Motivation Level",
        "StressLevel": "Stress Level"
    }

    metric = request.form.get("metric", "FinalGrade")
    aggregation = request.form.get("aggregation", "mean")

    if aggregation == "median":
        trend_df = df_t.groupby(["Period", "HighPerformer"])[metric].median().reset_index()
    elif aggregation == "sum":
        trend_df = df_t.groupby(["Period", "HighPerformer"])[metric].sum().reset_index()
    else:
        trend_df = df_t.groupby(["Period", "HighPerformer"])[metric].mean().reset_index()

    fig = px.line(
        trend_df,
        x="Period",
        y=metric,
        color="HighPerformer",
        markers=True,
        title=f"{METRICS[metric]} Trend Over Time"
    )

    return render_template(
        "trends.html",
        fig=fig.to_html(full_html=False),
        metric=metric,
        aggregation=aggregation,
        metrics=METRICS
    )

# =========================
# LOCAL RUN (Render uses gunicorn)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

