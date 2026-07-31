import os
from configs import config_image
from PIL import Image


def merge_image(path_folder):
    os.makedirs(config_image.dir_image_search, exist_ok=True)

    for cate in config_image.categorys:
        dir_folder = os.path.join(path_folder, cate)
        cout = 1

        for image in os.listdir(dir_folder):
            name, ext = os.path.splitext(image)
            new_name = f"{cate}_{cout}{ext}"
            cout += 1

            input_path = os.path.join(dir_folder, image)
            output_path = os.path.join(
                config_image.dir_image_search,
                new_name
            )

            img = Image.open(input_path)
            img.save(output_path)


if __name__ == "__main__":
    path_folder = os.path.join(
        config_image.dir_image_processed,
        "train"
    )
    merge_image(path_folder)

    print("success")