import os
from configs import paths_common


"""data"""
# directory data raw for chat
dir_chat_raw = os.path.join(paths_common.dir_data_raw,"chat")

#directory vector embedding data processed chat
dir_chat_processed = os.path.join(paths_common.dir_data_processed,"chat","faiss_db")


"""model"""
#directory save local model embedding
model_path = os.path.join(paths_common.dir_model,"chat","e5-base")