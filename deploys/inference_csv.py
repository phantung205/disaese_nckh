import os
import pandas as pd
import joblib
from src_csv import preprocessing
from configs import config_csv



def load_model(model_name):
    model_path = os.path.join(config_csv.dir_csv_model,"{}.pkl".format(model_name))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def  model_from_dic(input_dict,model_name):
    model =  load_model(model_name)

    # chuyển từ dạng dic sang dataframe
    df = pd.DataFrame([input_dict])

    # clear data
    df = preprocessing.clean_raw_data(df,False)

    # predict
    prediction = int(model.predict(df)[0])

    #   Predict probability
    probas = model.predict_proba(df)[0]
    classes = model.classes_
    proba_dict = {
        str(cls): round(float(p) * 100, 2)
        for cls, p in zip(classes, probas)
    }
    return prediction, proba_dict


def model_from_file(file_path, model_name):
    # load model
    model = load_model(model_name)

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
    predictions = model.predict(df_clean)

    result = df_clean.copy()
    result["prediction"] = predictions

    probas = model.predict_proba(df_clean)
    classes = model.classes_

    for i, cls in enumerate(classes):
        result[f"proba_class_{cls}"] = (probas[:, i] * 100).round(2)
    return result


if __name__ == '__main__':
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
    prediction, proba_dict = model_from_dic(sample,"logistic")
    print(prediction ,proba_dict)

    # test file
    test_file = os.path.join(config_csv.dir_csv_processed, "x_test.csv")
    df_result = model_from_file(test_file, "logistic")
    print(df_result.head())


