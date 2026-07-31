import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from configs import config_image

class ImageSearchEngine:
    def __init__(self):
        # load model
        self.model = SentenceTransformer(config_image.name_model_search,device="cuda")

        # load db vector
        self.index = faiss.read_index(config_image.index_file)

        # load image mapping
        with open(config_image.image_mapping_file) as f:
            self.image_files = [line.strip() for line in f]

    # biến input thành vector
    def encode_query(self, query):
        # up dạng ảnh
        if isinstance(query,Image.Image):
            embedding = self.model.encode(query)

        # đường dẫn ảnh
        elif isinstance(query, str) and query.endswith((".jpg", ".jpeg", ".png")):
            image = Image.open(query).convert("RGB")
            embedding = self.model.encode(image)

        else:
            raise ValueError("Unsupported query type")

        embedding = np.array(embedding).astype("float32").reshape(1, -1)
        return embedding

    # function search
    def search(self,query , top_k=5):
        # vector input
        query_embedding = self.encode_query(query)

        # lấy ra các vector giống nhất , và id
        distances, indices = self.index.search(query_embedding, top_k)

        # lấy ra đường dẫn ảnh
        results = [self.image_files[i] for i in indices[0]]

        return results