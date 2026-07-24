import os
import pandas as pd
from src_csv import preprocessing
from configs import config_csv
from src_csv.shap import loader,explain



def load_pipeline_explainer(model_name):
    pipeline, explainer = loader.load_shap(model_name)
    return pipeline, explainer


def  shap_from_dic(input_dict,pipeline, explainer):

    # chuyển từ dạng dic sang dataframe
    df = pd.DataFrame([input_dict])

    # clear data
    df = preprocessing.clean_raw_data(df,False)

    # predict
    prediction = int(pipeline.predict(df)[0])

    #   Predict probability
    probas = pipeline.predict_proba(df)[0]
    classes = pipeline.classes_
    proba_dict = {
        str(cls): round(float(p) * 100, 2)
        for cls, p in zip(classes, probas)
    }

    # SHAP
    shap_result = explain.explain(
        input_dict=input_dict,
        pipeline=pipeline,
        explainer=explainer
    )

    return prediction, proba_dict, shap_result


def shap_from_file(file_path,pipeline, explainer):

    # load data
    if file_path.endswith(".csv"):
        try:
            df = pd.read_csv(file_path)
        except Exception:
            raise ValueError("can not load file this csv ")
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        try:
            df = pd.read_excel(file_path)
        except Exception:
            raise ValueError("can not load file this exel ")
    else:
        raise ValueError("Only CSV or Excel files are supported")

    # clear data
    df_clean = preprocessing.clean_raw_data(df,False)

    # prediction
    predictions = pipeline.predict(df_clean)

    result = df_clean.copy()
    result["prediction"] = predictions

    probas = pipeline.predict_proba(df_clean)
    classes = pipeline.classes_

    for i, cls in enumerate(classes):
        result[f"proba_class_{cls}"] = (probas[:, i] * 100).round(2)

    # ==========================
    # SHAP Impact
    # ==========================

    # Giải thích SHAP cho từng dòng
    for _, row in df_clean.iterrows():

        # Tính SHAP
        shap_result = explain.explain(
            input_dict=row.to_dict(),
            pipeline=pipeline,
            explainer=explainer
        )

        # Ghi SHAP vào DataFrame
        for item in shap_result:
            feature = item["feature"]

            result.loc[row.name, f"{feature} Impact (%)"] = item["impact_percent"]


    return result


if __name__ == "__main__":

    # =====================================
    # Load model + SHAP một lần
    # =====================================
    pipeline, explainer = load_pipeline_explainer("logistic")

    # =====================================
    # Test dự đoán 1 người
    # =====================================
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

    prediction, proba_dict, shap_result = shap_from_dic(
        sample,
        pipeline,
        explainer
    )

    print("=" * 80)
    print("SINGLE PREDICTION")
    print("=" * 80)

    print("Prediction :", prediction)
    print("Probability :", proba_dict)

    print("\nSHAP Result")

    print(
        f"{shap_result['feature']}"
        f"{shap_result['impact_percent']}%"
    )

    # =====================================
    # Test file
    # =====================================

    test_file = os.path.join(
        config_csv.dir_csv_processed,
        "x_test.csv"
    )

    df_result = shap_from_file(
        test_file,
        pipeline,
        explainer
    )

    print("\n")
    print("=" * 80)
    print("FILE RESULT")
    print("=" * 80)

    print(df_result.head())

    # Nếu muốn lưu thử ra Excel
    output_path = os.path.join(
        config_csv.dir_csv_processed,
        "result_shap2.xlsx"
    )

    df_result.to_excel(
        output_path,
        index=False
    )

    print(f"\nSaved: {output_path}")