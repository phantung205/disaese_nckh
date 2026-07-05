from configs import paths_common
from datetime import datetime
import os


upload_folder = paths_common.dir_uploads

def save_upload_image(image):
    filename = image.filename

    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_filename = f"{name}_{timestamp}_{ext}"

    input_path = os.path.join(upload_folder,new_filename)
    image.save(input_path)

    return input_path ,new_filename


