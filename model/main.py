from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager, MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.lang import Builder
from kivy.core.window import Window
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

Window.size = (1000, 700)

# Load model and data
model = joblib.load("student_performance_model.pkl")
df = pd.read_csv("cleaned_dataset.csv")

# Get feature columns (exclude target)
features = [col for col in df.columns if col != "FinalGrade"]

class DashboardScreen(MDScreen):
    def on_enter(self):
        if "FinalGrade" in df.columns:
            plt.figure(figsize=(6, 4))
            df["FinalGrade"].value_counts().sort_index().plot(kind="bar")
            plt.title("Distribution of Final Grades")
            plt.tight_layout()
            os.makedirs("assets", exist_ok=True)
            plt.savefig("assets/grade_distribution.png")
            plt.close()
            self.ids.chart_image.source = "assets/grade_distribution.png"

class PredictionScreen(MDScreen):
    def on_pre_enter(self):
        container = self.ids.input_fields
        if len(container.children) > 0:
            return
        for feat in features:
            txt = MDTextField(hint_text=f"Enter {feat}", id=f"field_{feat}", mode="rectangle")
            container.add_widget(txt)

    def predict_result(self):
        inputs = {}
        for feat in features:
            field = None
            for w in self.ids.input_fields.children:
                if w.hint_text == f"Enter {feat}":
                    field = w
                    break
            value = field.text.strip() if field else ""
            try:
                inputs[feat] = float(value)
            except ValueError:
                inputs[feat] = 0.0
        data = pd.DataFrame([inputs])
        pred = model.predict(data)[0]
        self.ids.result_label.text = f"Predicted Grade: {pred}"

class AnalysisScreen(MDScreen):
    def on_enter(self):
        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(8, 4))
            importances = model.feature_importances_
            plt.barh(features, importances)
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig("assets/feature_importance.png")
            plt.close()
            self.ids.importance_image.source = "assets/feature_importance.png"

class StudentApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Amber"
        self.theme_cls.theme_style = "Light"

        Builder.load_file("dashboard_screen.kv")
        Builder.load_file("prediction_screen.kv")
        Builder.load_file("analysis_screen.kv")

        sm = MDScreenManager()
        sm.add_widget(DashboardScreen(name="dashboard"))
        sm.add_widget(PredictionScreen(name="predict"))
        sm.add_widget(AnalysisScreen(name="analysis"))
        return sm

if __name__ == "__main__":
    StudentApp().run()
