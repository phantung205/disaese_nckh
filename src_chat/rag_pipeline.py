from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

import base64

from src_chat.llm import get_llm
from src_chat.prompt import get_prompt
from src_chat.retriever import get_retriever


def format_docs(docs: list[Document]):
    return "\n\n".join(
        f"[Tài liệu {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs[:4])   # lấy 4 chunk, không cắt
    )

# CHAT TEXT BÌNH THƯỜNG
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


# CHAT + IMAGE
def answer_with_image(question: str,image_bytes: bytes, mime_type: str):

    retriever = get_retriever()

    if not question.strip():
        question = "hãy phân tích kết quả trong phiếu cho tôi , và đưa ra lời khuyên cho chỉ số ảnh hưởng đến kết quả"

    # RAG
    docs = retriever.invoke(question)

    context = format_docs(docs)

    # Image → Base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = f"""
Bạn là trợ lý AI hỗ trợ cung cấp thông tin về bệnh tiểu đường.

Bạn nhận được một hình ảnh và một câu hỏi.

QUY TẮC:

- Đọc thông tin trực tiếp từ hình ảnh.
- Không tự bịa thông tin không nhìn thấy.
- Sử dụng CONTEXT để giải thích kiến thức y khoa.
- Nếu không đủ thông tin, hãy nói rõ.
- Trả lời bằng tiếng Việt.
- Không tự khẳng định người dùng mắc bệnh.
- Giữ nguyên các giá trị số đọc được từ ảnh.

CONTEXT:
{context}

CÂU HỎI:
{question}

Hãy phân tích hình ảnh và trả lời câu hỏi.
"""

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{image_base64}"
                }
            }
        ]
    )

    response = get_llm().invoke([message])

    return response.content