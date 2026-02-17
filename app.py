import streamlit as st
import os
from utils import (
    init_session,
    load_existing_data,
    extract_text,
    add_document,
    search_query,
    generate_response
)

st.set_page_config(page_title="NeuroVault 2.0", layout="wide")

init_session()
load_existing_data()

os.makedirs("storage", exist_ok=True)

st.title("🧠 NeuroVault 2.0 - Chunked Persistent RAG System")

# =========================
# FILE UPLOAD
# =========================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("storage", file.name)

        if not os.path.exists(file_path):
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            text = extract_text(file)
            add_document(text, file.name)

            st.success(f"{file.name} indexed with chunk-based RAG!")

# =========================
# CHAT SECTION
# =========================

st.subheader("💬 Ask NeuroVault")

question = st.text_input("Type your question")

if st.button("Submit Question"):

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:
        relevant_chunks = search_query(question)

        if relevant_chunks is None:
            st.warning("No documents indexed yet.")

        else:
            prompt = f"""
You are an intelligent AI assistant.

Answer ONLY using the information provided below.
If answer is not present, say "Information not found in documents."

DOCUMENT CONTEXT:
{relevant_chunks}

QUESTION:
{question}

Provide a clear and accurate answer.
"""

            response = generate_response(prompt)

            st.subheader("🧠 NeuroVault Response")
            st.write(response)
