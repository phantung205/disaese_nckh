from src_chat.rag_pipeline import build_rag_chain, answer_with_image


# Build chain 1 lần
rag_chain = build_rag_chain()


def ask(question: str, image=None):

    # Không có ảnh → giữ nguyên cách chat hiện tại
    if image is None:
        return rag_chain.invoke(question)

    # Có ảnh → xử lý ảnh + RAG
    image_bytes = image.read()

    return answer_with_image(
        question=question,
        image_bytes=image_bytes,
        mime_type=image.mimetype
    )