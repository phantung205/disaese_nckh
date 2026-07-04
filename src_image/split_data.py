import os
import shutil
from configs import config_image
import random

random.seed(42)

def create_folders():
    for split in config_image.splits:
        for cls in config_image.categorys:
            path_file = os.path.join(config_image.dir_image_processed, split, cls)
            os.makedirs(path_file, exist_ok=True)

def split_data():
    for name_cls in os.listdir(config_image.dir_image_raw):
        path_cls = os.path.join(config_image.dir_image_raw, name_cls)

        images = [f for f in os.listdir(path_cls)
                  if f.endswith(('.png','.jpg','.jpeg'))]

        random.shuffle(images)

        total = len(images)
        train_end = int(total * config_image.train_ratio)

        train_imgs = images[:train_end]
        val_images = images[train_end:]

        for img_list, split in zip([train_imgs, val_images], config_image.splits):
            image_dir = os.path.join(config_image.data_processed_dir, split, name_cls)

            for img in img_list:
                src = os.path.join(path_cls, img)
                dst = os.path.join(image_dir, img)
                shutil.copy(src, dst)


if __name__ == '__main__':
    create_folders()
    split_data()