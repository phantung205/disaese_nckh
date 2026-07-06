from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from configs import config_chat
import os
from langchain_experimental.text_splitter import SemanticChunker

# model embedding text
embeddings = HuggingFaceEmbeddings(
    model_name=config_chat.model_path,
    encode_kwargs={"normalize_embeddings": True}
)

# Khởi tạo Semantic Chunker để chia văn bản theo ngữ nghĩa
loader = DirectoryLoader(
    path=config_chat.dir_chat_raw,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)
docs = loader.load()


# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap = 80,
#     add_start_index=True,
#     strip_whitespace=True,
#     separators=[
#         "\n\n",
#         "\n",
#         ". ",
#         "? ",
#         "! ",
#         "; ",
#         ", ",
#         " ",
#         ""
#     ]
# )

# Chia documents thành các text chunks
text_splitter = SemanticChunker(
    embeddings = embeddings,
    breakpoint_threshold_amount=0.65,
)

# chia dữ liệu thành các chunker
splits = text_splitter.split_documents(docs)


# lưu dữ liệu các chunker và vector db
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings,
    distance_strategy=DistanceStrategy.COSINE
)
os.makedirs(config_chat.dir_chat_processed, exist_ok=True)
vectorstore.save_local(config_chat.dir_chat_processed)
