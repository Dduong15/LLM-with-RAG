import streamlit as st
import os

from langchain_community.llms import Ollama
from document_loader import load_documents_into_database

from models import get_list_of_models

from llm import getStreamingChain


EMBEDDING_MODEL = "nomic-embed-text"
PATH = "E:/pdf"


st.title("Local LLM with RAG 📚")

if "list_of_models" not in st.session_state:
    st.session_state["list_of_models"] = get_list_of_models()

selected_model = st.sidebar.selectbox(
    "Chọn 1 model:", st.session_state["list_of_models"]
)

if st.session_state.get("ollama_model") != selected_model:
    st.session_state["ollama_model"] = selected_model
    st.session_state["llm"] = Ollama(model=selected_model)


# Folder selection
folder_path = st.sidebar.text_input("Nhập đường dẫn thư mục:", PATH)

if folder_path:
    if not os.path.isdir(folder_path):
        st.error(
            "Đường dẫn đã cung cấp không phải là thư mục hợp lệ. Vui lòng nhập một đường dẫn thư mục hợp lệ."
        )
    else:
        if st.sidebar.button("Chỉ mục tài liệu"):
            if "db" not in st.session_state:
                with st.spinner(
                    "Tạo embedding và tải tài liệu vào Chroma"
                ):
                    st.session_state["db"] = load_documents_into_database(
                        EMBEDDING_MODEL, folder_path
                    )
                st.info("Sẵn sàng để trả lời các câu hỏi!")
else:
    st.warning("Vui lòng nhập đường dẫn thư mục để tải tài liệu vào cơ sở dữ liệu")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = getStreamingChain(
            prompt,
            st.session_state.messages,
            st.session_state["llm"],
            st.session_state["db"],
        )
        response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
