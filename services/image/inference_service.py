from deploys import inference_image
from configs import config_image

_model = None
_device = None

def predict_image(image_path):
    model, device = load_model()

    result = inference_image.predict_image(
        model=model,
        device=device,
        image_path=image_path
    )

    return result

def load_model():
    global _model, _device

    if _model is None:
        _model, _device = inference_image.load_model(
            config_image.path_best_model
        )

    return _model, _device