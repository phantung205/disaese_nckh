import os

from configs import config_chat
from langchain_community.vectorstores import FAISS

from src_chat.embedding import get_embeddings

db_path = config_chat.dir_chat_processed

if not os.path.exists(db_path):
    raise ValueError("Chưa có FAISS DB. Hãy chạy vector_pipeline.py.")

embeddings = get_embeddings()

# Load FAISS 1 lần
vectorstore = FAISS.load_local(
    db_path,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3
    }
)


def get_retriever():
    return retriever