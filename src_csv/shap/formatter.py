import numpy as np


def format_shap_result(
    df,
    shap_values,
    feature_names
):

    values = shap_values.values

    if values.ndim == 2:
        values = values[0]

    elif values.ndim == 3:
        values = values[0, :, 1]

    else:
        raise ValueError(
            f"Shape không hỗ trợ: {values.shape}"
        )

    if len(values) != len(feature_names):
        raise ValueError(
            f"SHAP values: {len(values)}, "
            f"features: {len(feature_names)}"
        )

    # =====================================
    # GỘP SHAP VỀ FEATURE GỐC
    # =====================================

    grouped = {}

    for feature, shap_value in zip(
        feature_names,
        values
    ):

        # bỏ prefix của ColumnTransformer
        feature = feature.split("__", 1)[-1]

        # xác định feature gốc
        if feature.startswith("gender_"):
            original_feature = "gender"

        elif feature.startswith("smoking_history_"):
            original_feature = "smoking_history"

        else:
            original_feature = feature

        if original_feature not in grouped:
            grouped[original_feature] = 0.0

        grouped[original_feature] += float(shap_value)

    # =====================================
    # TÍNH IMPACT
    # =====================================

    total = sum(
        abs(value)
        for value in grouped.values()
    )

    result = []

    for feature, shap_value in grouped.items():

        impact = (
            abs(shap_value) / total * 100
            if total > 0
            else 0
        )

        value = df.iloc[0][feature]

        result.append({
            "feature": feature,
            "value": value,
            "shap": round(shap_value, 4),
            "impact_percent": round(impact, 2),
            "direction": (
                "increase"
                if shap_value >= 0
                else "decrease"
            )
        })

    result.sort(
        key=lambda x: x["impact_percent"],
        reverse=True
    )

    return result