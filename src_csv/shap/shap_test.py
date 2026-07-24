from src_csv.shap.explain import explain
from src_csv.shap.loader import load_shap

sample = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}

pipeline, explainer = load_shap("logistic")
result = explain(sample,pipeline, explainer)

print("=" * 90)
print("AI EXPLAIN RESULT")
print("=" * 90)

print(
    f"{'Feature':<30}"
    f"{'Value':>10}"
    f"{'SHAP':>12}"
    f"{'Impact':>12}"
    f"{'Direction':>12}"
)

print("-" * 90)

for item in result:

    direction = (
        "Increase ↑"
        if item["direction"] == "increase"
        else "Decrease ↓"
    )

    print(
        f"{item['feature']:<30}"
        f"{item['value']:>10.2f}"
        f"{item['shap']:>12.4f}"
        f"{item['impact_percent']:>11.2f}%"
        f"{direction:>12}"
    )