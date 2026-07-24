from deploys import inference_shap


# Lưu các model đã load
loaded_models = {}


def get_model(model_name):
    """
    Lấy model theo tên.

    Nếu model chưa được load thì load một lần.
    Nếu đã load rồi thì lấy từ RAM.
    """

    # Model chưa có trong cache
    if model_name not in loaded_models:

        print(f"Loading {model_name}...")

        pipeline,explainer= inference_shap.load_pipeline_explainer(model_name)

        loaded_models[model_name] = (pipeline,explainer)

        print(f"{model_name} loaded.")

    # Trả về model đã lưu
    return loaded_models[model_name]