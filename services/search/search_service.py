from configs import paths_common
import os
from datetime import datetime
from deploys import inference_search


uploads = paths_common.dir_uploads

def search_image(file,model,k_top=5):
    image_name = file.filename


    name,ext = os.path.splitext(image_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_name = f"{name}_{timestamp}{ext}"

    save_path = os.path.join(uploads,new_name)
    file.save(save_path)

    results = inference_search.search_image(save_path,model,k_top)

    results = [
        os.path.basename(path)
        for path in results
    ]

    return results, new_name