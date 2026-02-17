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
    get_dashboard_stats,
    get_important_files,
    check_similarity
)

st.set_page_config(page_title="NeuroVault AI", layout="wide")

# =========================
# INITIALIZE SYSTEM
# =========================

init_session()
load_existing_data()
os.makedirs("storage", exist_ok=True)

st.title("🧠 NeuroVault - Persistent AI Memory System")

# =========================
# FILE UPLOAD SECTION
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
# SIDEBAR DASHBOARD
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

    if st.sidebar.button("Delete File"):
        delete_document(selected)
        st.success("File deleted successfully!")
        st.rerun()

# =========================
# QUICK ACTION BUTTONS
# =========================

st.markdown("## ⚡ Quick Actions")

col1, col2, col3 = st.columns(3)

# Important Files
with col1:
    if st.button("📌 Important Files"):
        if total_files == 0:
            st.warning("No files uploaded.")
        else:
            important = get_important_files()
            for file in important:
                st.write(
                    f"{file['filename']} → Score: {file['importance_score']}"
                )

# Duplicate Check
with col2:
    if st.button("🔍 Duplicate Check"):
        if total_files < 2:
            st.warning("Upload at least 2 files.")
        else:
            sims = check_similarity()
            found = False
            for file1, file2, sim in sims:
                if sim > 0.80:
                    st.write(
                        f"{file1} and {file2} are {sim:.2f} similar"
                    )
                    found = True
            if not found:
                st.info("No highly similar documents found.")

# File Count
with col3:
    if st.button("📊 File Count"):
        st.info(f"You have uploaded {total_files} files.")

st.markdown("---")

# =========================
# CHAT SECTION
# =========================

st.subheader("💬 Ask NeuroVault")

question = st.text_input("Type your question")

if st.button("Submit Question"):
    if total_files == 0:
        st.warning("Upload at least one file first.")
    elif question.strip() == "":
        st.warning("Please enter a question.")
    else:
        relevant_docs = search_query(question)

        if relevant_docs is None:
            st.warning("No relevant document found.")
        else:
            prompt = f"""
You are an intelligent AI assistant.

Based on the following documents:

{relevant_docs}

Answer this question clearly and concisely:

{question}
"""

            response = generate_response(prompt)

            st.subheader("🧠 NeuroVault Response")
            st.write(response)
