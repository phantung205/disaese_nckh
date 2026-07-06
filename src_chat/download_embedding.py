from sentence_transformers import SentenceTransformer
from configs import config_chat

model = SentenceTransformer("intfloat/multilingual-e5-base")

model.save(config_chat.model_path)

print("Download thành công!")