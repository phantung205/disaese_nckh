from langchain_ollama import ChatOllama

# Load 1 lần
llm = ChatOllama(
    model="qwen3-vl:4b",
    temperature=0,
    num_ctx=4096,
    num_predict=512
)
# load model llm
def get_llm():
    return llm