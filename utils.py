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
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200


# =============================
# INITIALIZATION
# =============================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            chunk_text TEXT
        )
    """)

    conn.commit()
    conn.close()


def init_session():
    if "embed_model" not in st.session_state:
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    if "index" not in st.session_state:
        if os.path.exists(INDEX_PATH):
            st.session_state.index = faiss.read_index(INDEX_PATH)
        else:
            st.session_state.index = faiss.IndexFlatL2(EMBED_DIM)

    if "chunk_texts" not in st.session_state:
        st.session_state.chunk_texts = []

    if "model" not in st.session_state:
        initialize_ibm_model()


# =============================
# CHUNKING FUNCTION
# =============================

def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# =============================
# LOAD EXISTING DATA
# =============================

def load_existing_data():
    init_session()
    init_db()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT chunk_text FROM chunks")
    rows = c.fetchall()
    conn.close()

    st.session_state.chunk_texts = [row[0] for row in rows]

    rebuild_index()


def rebuild_index():
    if len(st.session_state.chunk_texts) == 0:
        st.session_state.index = faiss.IndexFlatL2(EMBED_DIM)
        return

    embeddings = st.session_state.embed_model.encode(
        st.session_state.chunk_texts
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
        params={"max_new_tokens": 400, "temperature": 0.2},
        credentials={"apikey": api_key, "url": url},
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

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


# =============================
# ADD DOCUMENT (Chunk Based)
# =============================

def add_document(text, filename):
    init_db()

    chunks = chunk_text(text)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for chunk in chunks:
        c.execute("""
            INSERT INTO chunks (filename, chunk_text)
            VALUES (?, ?)
        """, (filename, chunk))

    conn.commit()
    conn.close()

    load_existing_data()


# =============================
# SEARCH (Top 5 Chunks)
# =============================

def search_query(query):
    if len(st.session_state.chunk_texts) == 0:
        return None

    query_vector = st.session_state.embed_model.encode([query])
    D, I = st.session_state.index.search(
        np.array(query_vector).astype("float32"), k=5
    )

    relevant_chunks = ""

    for idx in I[0]:
        if idx < len(st.session_state.chunk_texts):
            relevant_chunks += (
                st.session_state.chunk_texts[idx] + "\n\n"
            )

    return relevant_chunks
