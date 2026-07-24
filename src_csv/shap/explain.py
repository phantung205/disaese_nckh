import pandas as pd
from src_csv import preprocessing
from src_csv.shap.formatter import format_shap_result


def explain(input_dict,pipeline, explainer):

    df = pd.DataFrame([input_dict])

    df = preprocessing.clean_raw_data(df,False)

    sample = pipeline.named_steps["preprocessor"].transform(df)

    shap_values = explainer(sample)

    return format_shap_result(df,shap_values)