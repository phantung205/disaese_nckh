from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src_chat.llm import get_llm
from src_chat.prompt import get_prompt
from src_chat.retriever import get_retriever


def format_docs(docs: list[Document]):

    return "\n\n".join(
        f"[Tài liệu {i + 1}]\n{doc.page_content[:700]}"
        for i, doc in enumerate(docs)
    )


def build_rag_chain():

    rag_chain = (
        {
            "context": get_retriever() | format_docs,
            "question": RunnablePassthrough()
        }
        | get_prompt()
        | get_llm()
        | StrOutputParser()
    )

    return rag_chain