import pandas as pd

from src_csv import preprocessing
from src_csv.shap.formatter import format_shap_result


def explain(input_dict, pipeline, explainer):

    df = pd.DataFrame([input_dict])

    df = preprocessing.clean_raw_data(
        df,
        False
    )

    preprocessor = pipeline.named_steps["preprocessor"]

    sample = preprocessor.transform(df)

    shap_values = explainer(sample)

    feature_names = preprocessor.get_feature_names_out()

    return format_shap_result(
        df,
        shap_values,
        feature_names
    )