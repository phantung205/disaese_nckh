from deploys import inference_csv


model = None
preprocessor = None
device = None


def load_model_nn():
    global  model
    global preprocessor
    global device
    if model is None or preprocessor is None or device is None:
        model, preprocessor, device = inference_csv.load_nn_model()

    return model,preprocessor,device