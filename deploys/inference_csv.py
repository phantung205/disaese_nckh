import os
import pandas as pd
import joblib
from src_csv import preprocessing
from configs import config_csv
import torch
from src_csv.neron.neural_network import DiabetesNN
import joblib

def load_model(model_name):
    model_path = os.path.join(config_csv.dir_csv_model,"{}.pkl".format(model_name))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)

def load_nn_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #  load tiền sử lý
    preprocessor = joblib.load(
        os.path.join(
            config_csv.dir_csv_model,
            "nn_preprocessor.pkl"
        )
    )

    #load checkpoint model
    checkpoint = torch.load(config_csv.path_best_model,map_location=device)

    input_dim = len(preprocessor.get_feature_names_out())
    model = DiabetesNN(input_dim=input_dim).to(device)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model, preprocessor, device

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


def nn_from_dic(input_dict,model, preprocessor, device):
    df = pd.DataFrame([input_dict])

    # clear dữ liệu
    df = preprocessing.clean_raw_data(df,False)

    #preprocessing
    x  = preprocessor.transform(df)

    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        probability = torch.sigmoid(logits).item()

    prediction = int(probability >= 0.45)

    proba_dict = {
        "0": round((1 - probability) * 100, 2),
        "1": round(probability * 100, 2)
    }

    return prediction, proba_dict


def nn_from_file(file_path,model, preprocessor, device):

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

    # preprocessing
    x = preprocessor.transform(df)

    # DataFrame / Series -> NumPy
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()

    # NumPy -> Tensor
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    predictions = (probabilities >= 0.45).astype(int)

    result = df_clean.copy()

    result["prediction"] = predictions

    result["proba_class_0"] = ((1 - probabilities) * 100).round(2)

    result["proba_class_1"] = (probabilities * 100).round(2)

    return result

if __name__ == '__main__':
    model, preprocessor, device = load_nn_model()

    sample = {
        "age": 45,
        "bmi": 28.5,
        "HbA1c_level": 6.2,
        "blood_glucose_level": 145,
        "hypertension": 1,
        "heart_disease": 0,
        "gender": "female",
        "smoking_history": "not current"
    }
    prediction, proba_dict = nn_from_dic(sample,model, preprocessor, device)
    print(prediction ,proba_dict)

    prediction, proba_dict = model_from_dic(sample,"logistic")
    print(prediction, proba_dict)

    # test file
    test_file = os.path.join(config_csv.dir_csv_processed, "x_test.csv")
    df_result = nn_from_file(test_file,model, preprocessor, device)
    print(df_result.head())


