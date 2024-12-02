import os

file_path = "E:/local-LLM-with-RAG-main/local-LLM-with-RAG-main/Bai4-GuiSV.pdf"

if os.path.exists(file_path):
    print("Tệp tồn tại.")
else:
    print("Tệp không tồn tại.")

from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader(file_path)
document = pdf_loader.load()
print(document[:500])  # In ra văn bản đầu tiên từ tài liệu