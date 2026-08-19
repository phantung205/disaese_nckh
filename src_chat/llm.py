from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="medgemma:4b",
    temperature=0,
    num_ctx=4096,
    num_predict=1024,
    think=False
)


def get_llm():
    return llm