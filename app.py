import streamlit as st
from utils import (
    extract_text,
    add_to_index,
    search_query,
    generate_response,
    get_important_files,
    check_similarity,
    get_dashboard_stats,
    generate_timeline,
    init_session
)

st.set_page_config(page_title="NeuroVault AI", layout="wide")

# =========================
# INIT SESSION
# =========================

init_session()

st.title("🧠 NeuroVault - AI Storage Intelligence System")

# =========================
# FILE UPLOAD
# =========================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# Detect removed files
current_filenames = [file.name for file in uploaded_files] if uploaded_files else []

stored_filenames = [file["filename"] for file in st.session_state.metadata]

# If files removed from uploader → remove from memory
for stored in stored_filenames:
    if stored not in current_filenames:
        index_to_remove = next(
            i for i, file in enumerate(st.session_state.metadata)
            if file["filename"] == stored
        )

        st.session_state.metadata.pop(index_to_remove)
        st.session_state.documents.pop(index_to_remove)

# Rebuild FAISS index if files removed
if len(st.session_state.documents) > 0:
    import faiss
    import numpy as np

    st.session_state.index = faiss.IndexFlatL2(384)

    embeddings = st.session_state.embed_model.encode(
        st.session_state.documents
    )

    st.session_state.index.add(np.array(embeddings).astype("float32"))

# Add new files
if uploaded_files:
    for file in uploaded_files:
        if file.name not in stored_filenames:
            text, pages = extract_text(file)
            add_to_index(text, file.name, pages)

    st.success("Files indexed successfully!")

# =========================
# DASHBOARD (AFTER UPLOAD)
# =========================

st.sidebar.header("📊 Storage Dashboard")

total_files, total_pages, total_words = get_dashboard_stats()

st.sidebar.metric("Total Files", total_files)
st.sidebar.metric("Total Pages", total_pages)
st.sidebar.metric("Total Words Indexed", total_words)

st.markdown("---")

# =========================
# QUICK ACTIONS
# =========================

col1, col2, col3, col4, col5 = st.columns(5)

# Important Files
with col1:
    if st.button("📌 Important Files"):
        important = get_important_files()
        if total_files == 0:
            st.warning("No files uploaded.")
        else:
            for file in important:
                st.write(f"{file['filename']} → Score: {file['importance_score']}")

# Duplicate Check
with col2:
    if st.button("🔍 Duplicate Check"):
        sims = check_similarity()
        found = False

        for file1, file2, sim in sims:
            if sim > 0.75:
                st.write(f"{file1} and {file2} are {sim:.2f} similar")
                found = True

        if not found:
            st.info("No highly similar files detected.")

# File Count
with col3:
    if st.button("📊 File Count"):
        st.write(f"You have uploaded {total_files} files.")

# Summarize Latest
with col4:
    if st.button("📝 Summarize Latest"):
        if total_files == 0:
            st.warning("No files uploaded.")
        else:
            latest = st.session_state.metadata[-1]
            prompt = f"Summarize this document:\n\n{latest['content'][:2000]}"
            summary = generate_response(prompt)
            st.write(summary)

# Memory Timeline
with col5:
    if st.button("📈 Memory Timeline"):
        timeline = generate_timeline()

        if not timeline:
            st.info("No version evolution detected.")
        else:
            for file1, file2, evolution in timeline:
                st.subheader(f"{file1} → {file2}")
                st.write(evolution)

st.markdown("---")

# =========================
# CHAT
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

        prompt = f"""
        Based on the following document:

        {relevant_doc}

        Answer this question:
        {question}
        """

        response = generate_response(prompt)

        st.subheader("NeuroVault Response")
        st.write(response)
