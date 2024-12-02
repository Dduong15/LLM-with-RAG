from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from models import check_if_model_is_available
from document_loader import load_documents_into_database
import argparse
import sys

from llm import getChatChain


def main(llm_model_name: str, embedding_model_name: str, documents_path: str) -> None:
    # Kiểm tra xem các mô hình có sẵn không, nếu không thì thử tải chúng về.
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    # Tạo cơ sở dữ liệu từ các tài liệu
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = Ollama(model=llm_model_name)
    chat = getChatChain(llm, db)

    while True:
        try:
            user_input = input(
                "\n\nVui lòng nhập câu hỏi của bạn (hoặc gõ 'exit' để kết thúc): "
            )
            if user_input.lower() == "exit":
                break

            chat(user_input)
        except KeyboardInterrupt:
            break


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy local LLM với RAG sử dụng Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        default="mistral",
        help="Tên của mô hình LLM để sử dụng.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        default="nomic-embed-text",
        help="Tên của mô hình nhúng để sử dụng.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="Research",
        help="Đường dẫn đến thư mục chứa các tài liệu để tải lên.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.model, args.embedding_model, args.path)
