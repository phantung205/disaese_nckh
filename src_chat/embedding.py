from langchain_huggingface import HuggingFaceEmbeddings
from configs import config_chat

# Load 1 lần
embeddings = HuggingFaceEmbeddings(
    model_name=config_chat.model_path,
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# load model embedding
def get_embeddings():
    return embeddings