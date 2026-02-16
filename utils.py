import streamlit as st
import os
import sqlite3
import numpy as np
import PyPDF2
import faiss
from ibm_watsonx_ai.foundation_models import Model
from sentence_transformers import SentenceTransformer

# =============================
# DATABASE INITIALIZATION
# =============================

def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT,
            pages INTEGER,
            words INTEGER
        )
    """)

    conn.commit()
    conn.close()


# =============================
# SESSION INITIALIZATION
# =============================

def init_session():

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "metadata" not in st.session_state:
        st.session_state.metadata = []

    if "index" not in st.session_state:
        st.session_state.index = faiss.IndexFlatL2(384)

    if "embed_model" not in st.session_state:
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if "model" not in st.session_state:
        initialize_ibm_model()


# =============================
# LOAD EXISTING DATA (Persistent Memory)
# =============================

def load_existing_data():
    init_session()
    init_db()

    # Load FAISS index if exists
    if os.path.exists("faiss_index.index"):
        st.session_state.index = faiss.read_index("faiss_index.index")

    # Load stored documents from database
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT filename, content, pages, words FROM documents")
    rows = c.fetchall()
    conn.close()

    for row in rows:
        filename, content, pages, words = row

        if content not in st.session_state.documents:

            st.session_state.documents.append(content)

            st.session_state.metadata.append({
                "filename": filename,
                "importance_score": 0,
                "pages": pages,
                "words": words,
                "content": content
            })


# =============================
# IBM MODEL (SAFE LOAD)
# =============================

def initialize_ibm_model():
    try:
        api_key = st.secrets["IBM_API_KEY"]
        project_id = st.secrets["IBM_PROJECT_ID"]
        url = st.secrets["IBM_URL"]

        st.session_state.model = Model(
            model_id="ibm/granite-3-8b-instruct",
            params={
                "max_new_tokens": 300,
                "temperature": 0.3
            },
            credentials={
                "apikey": api_key,
                "url": url
            },
            project_id=project_id
        )

    except Exception:
        st.error("IBM Model Initialization Failed. Check Secrets.")
        st.stop()


def generate_response(prompt):
    init_session()
    response = st.session_state.model.generate(prompt)
    return response["results"][0]["generated_text"]


# =============================
# PDF EXTRACTION
# =============================

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    pages = len(reader.pages)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text, pages


# =============================
# ADD FILE (Persistent)
# =============================

def add_to_index(text, filename, pages):
    init_session()
    init_db()

    if text in st.session_state.documents:
        return

    # Add embedding
    embedding = st.session_state.embed_model.encode([text])
    st.session_state.index.add(np.array(embedding).astype("float32"))

    st.session_state.documents.append(text)

    score = 0
    if "final" in filename.lower():
        score += 2
    if "v2" in filename.lower() or "v3" in filename.lower():
        score += 1
    if len(text) > 2000:
        score += 1

    metadata_entry = {
        "filename": filename,
        "importance_score": score,
        "pages": pages,
        "words": len(text.split()),
        "content": text
    }

    st.session_state.metadata.append(metadata_entry)

    # Save to database
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO documents (filename, content, pages, words)
        VALUES (?, ?, ?, ?)
    """, (filename, text, pages, len(text.split())))
    conn.commit()
    conn.close()

    # Save FAISS index permanently
    faiss.write_index(st.session_state.index, "faiss_index.index")


# =============================
# SEARCH
# =============================

def search_query(query):
    init_session()

    if len(st.session_state.documents) == 0:
        return "No documents indexed yet."

    query_vector = st.session_state.embed_model.encode([query])

    D, I = st.session_state.index.search(
        np.array(query_vector).astype("float32"), k=1
    )

    return st.session_state.documents[I[0][0]]


# =============================
# IMPORTANT FILES
# =============================

def get_important_files():
    init_session()
    return sorted(
        st.session_state.metadata,
        key=lambda x: x["importance_score"],
        reverse=True
    )


# =============================
# DUPLICATE CHECK
# =============================

def check_similarity():
    init_session()
    similarities = []

    docs = st.session_state.documents
    embed_model = st.session_state.embed_model

    embeddings = embed_model.encode(docs)

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):

            sim = np.dot(
                embeddings[i],
                embeddings[j]
            )

            similarities.append(
                (
                    st.session_state.metadata[i]["filename"],
                    st.session_state.metadata[j]["filename"],
                    sim
                )
            )

    return similarities


# =============================
# DASHBOARD
# =============================

def get_dashboard_stats():
    init_session()

    total_files = len(st.session_state.metadata)
    total_pages = sum(file["pages"] for file in st.session_state.metadata)
    total_words = sum(file["words"] for file in st.session_state.metadata)

    return total_files, total_pages, total_words


# =============================
# MEMORY TIMELINE
# =============================

def generate_timeline():
    init_session()

    similarities = check_similarity()
    timeline = []

    for file1, file2, sim in similarities:
        if sim > 0.75:

            doc1 = next(
                item for item in st.session_state.metadata
                if item["filename"] == file1
            )

            doc2 = next(
                item for item in st.session_state.metadata
                if item["filename"] == file2
            )

            prompt = f"""
            Compare these two document versions.

            Document A:
            {doc1["content"][:1500]}

            Document B:
            {doc2["content"][:1500]}

            Explain how they evolved.
            """

            evolution = generate_response(prompt)
            timeline.append((file1, file2, evolution))

    return timeline
