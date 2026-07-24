import os
import joblib
import pandas as pd
import shap

from configs import config_csv
from src_csv import preprocessing


def load_pipeline(model_name):

    model_path = os.path.join(config_csv.dir_csv_model,f"{model_name}.pkl")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)

    return joblib.load(model_path)


def load_shap(model_name):

    pipeline = load_pipeline(model_name)

    preprocessor = pipeline.named_steps["preprocessor"]

    clf = pipeline.named_steps["clf"]

    x_train = pd.read_csv(
        os.path.join(config_csv.dir_csv_processed,"x_train.csv")
    )

    x_train = preprocessing.clean_raw_data(x_train,False)

    x_train = preprocessor.transform(x_train)

    if model_name == "logistic":
        explainer = shap.LinearExplainer(clf,x_train)

    elif model_name == "random_forest":
        explainer = shap.TreeExplainer(clf)

    elif model_name == "xgboost":
        explainer = shap.TreeExplainer(clf)

    elif model_name == "svm":
        background = shap.sample(x_train,50,random_state=42)
        explainer = shap.KernelExplainer(clf.predict_proba,background)

    else:
        raise ValueError("Model not support.")

    return pipeline, explainer