from configs import config_image
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose,RandomAffine,Resize,ToTensor,Normalize,RandomHorizontalFlip,RandomRotation,ColorJitter
import os
from PIL import Image

class DiabeticRetinopathyDataset(Dataset):
    def __init__(self,root,train=True,transform=None):
        if train:
            mode = "train"
        else:
            mode = "valid"

        root = os.path.join(root,mode)
        self.transform = transform
        self.categories = config_image.categorys

        self.images_paths = []
        self.labels = []

        for idx, category in enumerate(self.categories):
            data_file_path = os.path.join(root,category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path,file_name)
                self.images_paths.append(file_path)
                self.labels.append(idx)

    def __len__(self):
        return  len(self.labels)

    def __getitem__(self, item):
        image_path = self.images_paths[item]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.labels[item]
        return image, label

def dataloader(batch_size,image_size):
    train_transform = Compose([
        Resize((image_size, image_size)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(15),
        RandomAffine(
            degrees=10,
            translate=(0.03, 0.03),
            scale=(0.9, 1.1),
            shear=3
        ),
        ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.05,
            hue=0.02
        ),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    train_dataset = DiabeticRetinopathyDataset(config_image.dir_image_processed,train=True,transform=train_transform)

    val_transfrom = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    val_dataset = DiabeticRetinopathyDataset(config_image.dir_image_processed,train=False,transform=val_transfrom)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    return train_dataloader, test_dataloader

