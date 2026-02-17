import streamlit as st
import os
from utils import (
    init_session,
    load_existing_data,
    extract_text,
    add_document,
    delete_document,
    search_query,
    generate_response,
    get_dashboard_stats
)

st.set_page_config(page_title="NeuroVault AI", layout="wide")

init_session()
load_existing_data()

st.title("🧠 NeuroVault - Persistent AI Memory System")

os.makedirs("storage", exist_ok=True)

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

            text, pages = extract_text(file)
            success = add_document(text, file.name, pages)

            if success:
                st.success(f"{file.name} indexed successfully!")

# =========================
# DASHBOARD
# =========================

st.sidebar.header("📊 Storage Dashboard")

total_files, total_pages, total_words = get_dashboard_stats()

st.sidebar.metric("Total Files", total_files)
st.sidebar.metric("Total Pages", total_pages)
st.sidebar.metric("Total Words Indexed", total_words)

st.sidebar.markdown("---")

# =========================
# DELETE FILE OPTION
# =========================

if total_files > 0:
    st.sidebar.subheader("🗑 Delete File")
    filenames = [file["filename"] for file in st.session_state.metadata]
    selected = st.sidebar.selectbox("Select file", filenames)

    if st.sidebar.button("Delete"):
        delete_document(selected)
        st.success("File deleted successfully!")
        st.rerun()

# =========================
# CHAT SECTION
# =========================

st.subheader("💬 Ask NeuroVault")

question = st.text_input("Type your question")

if st.button("Submit"):
    if total_files == 0:
        st.warning("Upload at least one file first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        relevant_doc = search_query(question)

        if relevant_doc is None:
            st.warning("No relevant document found.")
        else:
            prompt = f"""
            Based on the following document:

            {relevant_doc[:3000]}

            Answer this question:
            {question}
            """

            response = generate_response(prompt)

            st.subheader("NeuroVault Response")
            st.write(response)
