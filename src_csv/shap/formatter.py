import numpy as np


def format_shap_result(df, shap_values):

    values = shap_values.values

    # Logistic
    if values.ndim == 2:
        values = values[0]

    # Random Forest / SVM
    elif values.ndim == 3:
        values = values[0, :, 1]      # lấy class 1 (diabetes)

    else:
        raise ValueError(
            f"Shape không hỗ trợ: {values.shape}"
        )

    scores = np.abs(values)

    impact = scores / scores.sum() * 100

    result = []

    for feature, value, score in zip(df.columns,values,impact):

        result.append({
            "feature": feature,
            "value": float(df.iloc[0][feature]),
            "shap": round(float(value),4),
            "impact_percent": round(float(score),2),
            "direction": (
                "increase"
                if value >= 0
                else "decrease"
            )

        })

    result.sort(
        key=lambda x:x["impact_percent"],
        reverse=True
    )

    return result