import argparse
from src_image.model import DiabeticRetinopathy
import cv2
import torch
from torchvision.transforms import Compose,ToTensor,Resize
from configs import config_image
import torch.nn as nn
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--image_path","-i",type=str,default="no.png")
    parser.add_argument("--checkpoint","-c",type=str,default=config_image.path_best_model)

    return parser.parse_args()


def main(args,image_path):

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiabeticRetinopathy().to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model"])
    else:
        print("no checkpoint found")
        exit(0)

    # load image
    ori_image = cv2.imread(image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

    # transfrom
    image = cv2.resize(image, (config_image.image_size, config_image.image_size))
    image = np.transpose(image, (2, 0, 1)) / 255.0

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    image = (image - mean) / std

    # add batch size
    image = image[None, :, :, :]
    image = torch.from_numpy(image).to(device).float()

    # turn on model eval
    model.eval()
    softmax = nn.Softmax()
    with torch.no_grad():
        output = model(image)
        print(output)
        probabirity = softmax(output)
        print(probabirity)

    # Extract the vector with the highest score
    max_idx = torch.argmax(probabirity)
    predicted_class = config_image.categorys[max_idx]
    print("the test image is abount {} with confident score of {:.4f}".format(predicted_class,
                                                                              probabirity[0, max_idx]))
    cv2.imshow("{} : {:.2f}%".format(predicted_class, probabirity[0, max_idx] * 100), ori_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = get_args()
    main(args=args,image_path = args.image_path)