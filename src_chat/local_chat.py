from src_chat.rag_pipeline import build_rag_chain


def main():

    rag_chain = build_rag_chain()

    print("=" * 60)
    print("Medical ChatBot")
    print("Nhập exit để thoát.")
    print("=" * 60)

    while True:

        question = input("\nBạn: ")

        if question.lower() in ["exit", "quit"]:

            break

        answer = rag_chain.invoke(question)

        print("\nBot:", answer)


if __name__ == "__main__":

    main()