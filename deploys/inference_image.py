import argparse
from src_image.model import DiabeticRetinopathy
import cv2
import torch
from torchvision.transforms import Compose,ToTensor,Resize
from configs import config_image
import torch.nn as nn
import numpy as np

def load_model(checkpoint_path):
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiabeticRetinopathy().to(device)

    checkpoint = torch.load(checkpoint_path,map_location=device)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    return model, device

def preprocessed_image(image_path,image_size):
    # đọc ảnh
    image = cv2.imread(image_path)

    # kiêm tra xem có đọc đc ảnh ko
    if image is None:
        raise ValueError("File tải lên không phải là ảnh hợp lệ.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (image_size,image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    image = (image - mean) / std

    # add batch size
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float()

    return image

def predict_image(model,device,image_path):
    image = preprocessed_image(image_path,config_image.image_size)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return {
        "class": config_image.categorys[predicted.item()],
        "confidence": float(confidence.item())
    }


if __name__ == '__main__':
    model,device = load_model(config_image.path_best_model)
    result = predict_image(model,device,"no.png")
    print(result)