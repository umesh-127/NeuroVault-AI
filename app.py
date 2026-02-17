import streamlit as st
import os
import sqlite3
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import Model

DB_PATH = "database.db"
INDEX_PATH = "faiss_index.index"
STORAGE_FOLDER = "storage"
EMBED_DIM = 384


# =============================
# INITIALIZATION
# =============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            content TEXT,
            pages INTEGER,
            words INTEGER
        )
    """)

    conn.commit()
    conn.close()


def init_session():
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if "documents" not in st.session_state:
        st.session_state.documents = []

    if "metadata" not in st.session_state:
        st.session_state.metadata = []

    if "index" not in st.session_state:
        if os.path.exists(INDEX_PATH):
            st.session_state.index = faiss.read_index(INDEX_PATH)
        else:
            st.session_state.index = faiss.IndexFlatL2(EMBED_DIM)

    if "model" not in st.session_state:
        initialize_ibm_model()


# =============================
# LOAD EXISTING MEMORY
# =============================

def load_existing_data():
    init_session()
    init_db()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT filename, content, pages, words FROM documents")
    rows = c.fetchall()
    conn.close()

    st.session_state.documents.clear()
    st.session_state.metadata.clear()

    for filename, content, pages, words in rows:
        st.session_state.documents.append(content)
        st.session_state.metadata.append({
            "filename": filename,
            "pages": pages,
            "words": words,
            "content": content
        })

    rebuild_faiss()


def rebuild_faiss():
    if len(st.session_state.documents) == 0:
        st.session_state.index = faiss.IndexFlatL2(EMBED_DIM)
        return

    embeddings = st.session_state.embed_model.encode(
        st.session_state.documents
    )

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, INDEX_PATH)

    st.session_state.index = index


# =============================
# IBM MODEL
# =============================

def initialize_ibm_model():
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


def generate_response(prompt):
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
# ADD DOCUMENT
# =============================

def add_document(text, filename, pages):
    init_db()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        c.execute("""
            INSERT INTO documents (filename, content, pages, words)
            VALUES (?, ?, ?, ?)
        """, (filename, text, pages, len(text.split())))
        conn.commit()
    except:
        conn.close()
        return False

    conn.close()
    load_existing_data()
    return True


# =============================
# DELETE DOCUMENT
# =============================

def delete_document(filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()

    file_path = os.path.join(STORAGE_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    load_existing_data()


# =============================
# SEARCH
# =============================

def search_query(query):
    if len(st.session_state.documents) == 0:
        return None

    query_vector = st.session_state.embed_model.encode([query])
    D, I = st.session_state.index.search(
        np.array(query_vector).astype("float32"), k=1
    )

    idx = I[0][0]
    return st.session_state.documents[idx]


# =============================
# DASHBOARD
# =============================

def get_dashboard_stats():
    total_files = len(st.session_state.metadata)
    total_pages = sum(file["pages"] for file in st.session_state.metadata)
    total_words = sum(file["words"] for file in st.session_state.metadata)

    return total_files, total_pages, total_words
