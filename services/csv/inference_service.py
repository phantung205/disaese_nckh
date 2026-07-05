from deploys import inference_csv

def predict_dict(data,model_name):
    prediction, proba = inference_csv.model_from_dic(data, model_name)
    return prediction, proba

def predict_file(input_path,model_name):
    df_result = inference_csv.model_from_file(input_path, model_name)

    return df_result