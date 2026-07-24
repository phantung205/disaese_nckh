from deploys import inference_csv,inference_shap
from services.csv import load_pipeline_explainer

def predict_dict_base(data,model_name):
    prediction, proba = inference_csv.model_from_dic(data, model_name)
    return prediction, proba

def predict_file_base(input_path,model_name):
    df_result = inference_csv.model_from_file(input_path, model_name)

    return df_result



def predict_dict_shap(data, model_name):
    pipeline, explainer = load_pipeline_explainer.get_model(model_name)

    prediction, proba_dict, shap_result = inference_shap.shap_from_dic(
        data,
        pipeline,
        explainer
    )

    return prediction, proba_dict, shap_result


def predict_file_shap(path_file, model_name):

    pipeline, explainer = load_pipeline_explainer.get_model(model_name)

    df_result = inference_shap.shap_from_file(
        path_file,
        pipeline,
        explainer
    )

    return df_result