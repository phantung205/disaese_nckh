from sentence_transformers import SentenceTransformer
from configs import config_image
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import faiss


# load model embedding image
def load_model():
    model = SentenceTransformer(config_image.name_model_search)
    return  model


# load image path
def load_image_paths():
    image_files = glob(os.path.join(config_image.dir_image_search,"*.jpg"))

    if len(image_files) == 0:
        raise ValueError(" No images found!")
    return image_files


# embbeding ảnh
def create_embeddings(model,image_file):

    embeddings = []
    batch_size = config_image.batch_size_search

    for i in tqdm(range(0,len(image_file),batch_size)):
        # lấy từng batch một
        batch_paths = image_file[i:i+batch_size]

        images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except:
                continue

        # embedding ảnh thành các vector
        batch_embeddings = model.encode(
            images,
            batch_size=len(images),
            show_progress_bar= False
        )
        embeddings.extend(batch_embeddings)

    return np.array(embeddings).astype("float32")

# create vector DB lưu chữ và tìm kiếm nhanh nhất
def build_faiss_index(embeddings):
    # lấy ra số lượng chiều dài vector
    dimension =  embeddings.shape[1]

    # chỉ tạo engine biết cách đo độ tương đồng bằng Inner Product
    index = faiss.IndexFlatIP(dimension)
    # gán id cho từng vector
    index = faiss.IndexIDMap(index)
    # tạo id cho ảnh
    ids = np.arange(len(embeddings))

    index.add_with_ids(embeddings, ids)

    return index

# save index mapping
def save_index(index,image_files):
    os.makedirs(config_image.dir_index, exist_ok=True)

    print("-----------------")
    # lưu các vector xuống ổ cứng
    faiss.write_index(index, config_image.index_file)

    print("------------------")
    with open(config_image.image_mapping_file, "w") as f:
        for path in image_files:
            f.write(path + "\n")


# main pipline
def run_embedding_pipeline():

    model = load_model()

    image_files = load_image_paths()

    embeddings = create_embeddings(model, image_files)

    index = build_faiss_index(embeddings)

    save_index(index, image_files)

if __name__ == "__main__":
    run_embedding_pipeline()