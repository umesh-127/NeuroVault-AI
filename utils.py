import streamlit as st
import numpy as np
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import Model


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
# ADD TO INDEX
# =============================

def add_to_index(text, filename, pages):

    if text in st.session_state.documents:
        return

    embedding = st.session_state.embed_model.encode([text])
    st.session_state.index.add(np.array(embedding).astype("float32"))

    score = 0
    if "final" in filename.lower():
        score += 2
    if "v2" in filename.lower() or "v3" in filename.lower():
        score += 1
    if len(text) > 2000:
        score += 1

    st.session_state.documents.append(text)

    st.session_state.metadata.append({
        "filename": filename,
        "importance_score": score,
        "pages": pages,
        "words": len(text.split()),
        "content": text
    })


# =============================
# SEARCH
# =============================

def search_query(query):

    if len(st.session_state.documents) == 0:
        return "No documents indexed yet."

    if st.session_state.index.ntotal == 0:
        return "Search index is empty."

    query_vector = st.session_state.embed_model.encode([query])

    D, I = st.session_state.index.search(
        np.array(query_vector).astype("float32"), k=1
    )

    idx = I[0][0]

    if idx >= len(st.session_state.documents):
        return "Search mismatch error."

    return st.session_state.documents[idx]


# =============================
# IMPORTANT FILES
# =============================

def get_important_files():
    return sorted(
        st.session_state.metadata,
        key=lambda x: x["importance_score"],
        reverse=True
    )


# =============================
# DUPLICATE CHECK
# =============================

def check_similarity():

    docs = st.session_state.documents

    if len(docs) < 2:
        return []

    embeddings = st.session_state.embed_model.encode(docs)

    similarities = []

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

    total_files = len(st.session_state.metadata)
    total_pages = sum(file["pages"] for file in st.session_state.metadata)
    total_words = sum(file["words"] for file in st.session_state.metadata)

    return total_files, total_pages, total_words


# =============================
# MEMORY TIMELINE
# =============================

def generate_timeline():

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
