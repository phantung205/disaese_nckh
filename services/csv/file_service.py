import os
from datetime import datetime
from configs import paths_common

upload_folder = paths_common.dir_uploads
result_folder = paths_common.dir_results


def save_upload_file(file):
    filename = file.filename

    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%S_%M_%H_%d_%m_%Y")
    new_filename = f"{name}_{timestamp}{ext}"
    input_path = os.path.join(upload_folder, new_filename)
    file.save(input_path)

    return input_path,name,timestamp

def save_prediction_result(df_result, name, timestamp):
    # lưu file kết quả
    output_filename = f"{name}_prediction_{timestamp}.csv"
    output_path = os.path.join(result_folder, output_filename)
    df_result.to_csv(output_path, index=False)

    return output_filename

def get_download_path(filename):
    return os.path.join(result_folder,filename)