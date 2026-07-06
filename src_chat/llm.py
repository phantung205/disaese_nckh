from langchain_ollama import ChatOllama

# Load 1 lần
llm = ChatOllama(
    model="qwen2:7b",
    temperature=0,
    num_ctx=512,
    num_predict=128
)

# load model llm
def get_llm():
    return llm