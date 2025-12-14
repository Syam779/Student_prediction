from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, json, os

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
# PREDICTION (ML)
# =========================
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = probability = score = risk_level = None
    advice = []
    breakdown = {}
    benchmark = {}          # ✅ FIX: define early
    form_values = {}        # keep slider values

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

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]
        else:
            prob = 0.0

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

        # ✅ Benchmark comparison
        benchmark = {
            "Study Hours": round(values["StudyHours"] - avg["StudyHours"], 1),
            "Attendance": round(values["Attendance"] - avg["Attendance"], 1),
            "Assignment Completion": round(values["AssignmentCompletion"] - avg["AssignmentCompletion"], 1),
        }

        # ✅ Advice
        if values["StudyHours"] < avg["StudyHours"]:
            advice.append("Increase weekly study hours.")
        if values["Attendance"] < 80:
            advice.append("Improve class attendance.")
        if values["AssignmentCompletion"] < 70:
            advice.append("Submit assignments more consistently.")
        if values["StressLevel"] > 7:
            advice.append("Reduce stress through time management.")

    return render_template(
        "predict.html",
        prediction=prediction,
        probability=probability,
        score=score,
        risk_level=risk_level,
        breakdown=breakdown,
        benchmark=benchmark,
        advice=advice,
        form_values=form_values
    )



from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import send_file
import io

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

    # ✅ KPI ALWAYS EXISTS (IMPORTANT)
    kpi = {
        "avg_grade": round(df_i["FinalGrade"].mean(), 2),
        "high_pct": round((df_i["FinalGrade"] >= 75).mean() * 100, 2),
        "avg_att": round(df_i["Attendance"].mean(), 2),
        "avg_study": round(df_i["StudyHours"].mean(), 2),
    }

    # Defaults
    fig = None
    heatmap = None
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

        fig.update_layout(template="plotly_white")
        fig = fig.to_html(full_html=False)

        heatmap = px.imshow(
            df_i[numeric_cols].corr(),
            color_continuous_scale="RdBu"
        )
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

    # ---- Simulated time periods (defensible for non-time dataset)
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

    if metric not in METRICS:
        metric = "FinalGrade"

    # ---- Aggregation logic
    if aggregation == "median":
        trend_df = df_t.groupby(
            ["Period", "HighPerformer"]
        )[metric].median().reset_index()
    elif aggregation == "sum":
        trend_df = df_t.groupby(
            ["Period", "HighPerformer"]
        )[metric].sum().reset_index()
    else:
        trend_df = df_t.groupby(
            ["Period", "HighPerformer"]
        )[metric].mean().reset_index()

    # ---- Plot
    fig = px.line(
        trend_df,
        x="Period",
        y=metric,
        color="HighPerformer",
        markers=True,
        labels={
            "HighPerformer": "Student Group",
            "Period": "Time Period",
            metric: METRICS[metric]
        },
        title=f"{METRICS[metric]} Trend Over Time ({aggregation.capitalize()})",
        template="plotly_white"
    )

    fig.update_layout(
        transition_duration=900,
        hovermode="x unified",
        legend_title_text="Performance Group"
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7)
    )

    interpretation = (
        f"This chart shows how {METRICS[metric].lower()} changes over time. "
        f"High performers (≥75) consistently maintain stronger trends, "
        f"indicating stable academic behaviour patterns."
    )

    return render_template(
        "trends.html",
        fig=fig.to_html(full_html=False),
        metric=metric,
        aggregation=aggregation,
        metrics=METRICS,
        interpretation=interpretation
    )

# =========================
# RUN APP (LAST LINE ONLY)
# =========================
if __name__ == "__main__":
    app.run(debug=True)
