from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Tải tài liệu từ thư mục đã chỉ định vào cơ sở dữ liệu Chroma sau khi chia nhỏ văn bản thành các đoạn.

    Trả về: Chroma: Cơ sở dữ liệu Chroma với các tài liệu đã được tải lên.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Đang tạo embedding và tải tài liệu vào Chroma.")
    db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
    )
    return db


def load_documents(path: str) -> List[Document]:
    """
    Tải tài liệu từ đường dẫn thư mục đã chỉ định.

    Hàm này hỗ trợ tải các tài liệu PDF, Markdown và HTML bằng cách sử dụng các loader khác nhau cho mỗi loại tệp. Nó kiểm tra xem đường dẫn đã cung cấp có tồn tại không và sẽ ném ra lỗi FileNotFoundError nếu không tồn tại. Sau đó, nó lặp qua các loại tệp được hỗ trợ và sử dụng loader tương ứng để tải tài liệu vào danh sách.

    Tham số:

    path (str): Đường dẫn tới thư mục chứa các tài liệu cần tải.
    Kết quả trả về:

    List[Document]: Một danh sách các tài liệu đã tải.
    Ngoại lệ:

    FileNotFoundError: Nếu đường dẫn chỉ định không tồn tại.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Đường dẫn được chỉ định không tồn tại: {path}")

    loaders = {
        ".pdf": DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls = PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
        ),
        ".md": DirectoryLoader(
            path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True,
        ),
    }

    docs = []
    for file_type, loader in loaders.items():
        print(f"Loading {file_type} files")
        docs.extend(loader.load())
    return docs
