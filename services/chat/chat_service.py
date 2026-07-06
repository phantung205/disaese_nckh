from src_chat.rag_pipeline import build_rag_chain

# Build chain 1 lần
rag_chain = build_rag_chain()


def ask(question: str):

    return rag_chain.invoke(question)