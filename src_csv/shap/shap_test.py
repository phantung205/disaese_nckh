from src_csv.shap.explain import explain
from src_csv.shap.loader import load_shap

sample = {
    "age": 45,
    "bmi": 28.5,
    "HbA1c_level": 6.2,
    "blood_glucose_level": 145,
    "hypertension": 1,
    "heart_disease": 0,
    "gender": "female",
    "smoking_history": "never"
}

pipeline, explainer = load_shap("random_forest")
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

    value = item["value"]

    if isinstance(value, (int, float)):
        value_str = f"{value:.2f}"
    else:
        value_str = str(value)

    print(
        f"{item['feature']:<30}"
        f"{value_str:>10}"
        f"{item['shap']:>12.4f}"
        f"{item['impact_percent']:>11.2f}%"
        f"{direction:>12}"
    )