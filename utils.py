from ibm_watsonx_ai.foundation_models import Model
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2

# 🔐 Replace with your new API key
api_key = "EeHCPhVryxRZ9znh5IAXklko6STOjGtdyItZop2xo8rK"
project_id = "d59dd38a-f7a3-450d-9bc7-c15517be643e"
url = "https://us-south.ml.cloud.ibm.com"

model = Model(
    model_id="ibm/granite-3-8b-instruct",
    params={"max_new_tokens": 300, "temperature": 0.3},
    credentials={"apikey": api_key, "url": url},
    project_id=project_id
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ===== STORAGE =====
documents = []
metadata = []

# ===== MODEL FUNCTIONS =====

def generate_response(prompt):
    response = model.generate(prompt)
    return response["results"][0]["generated_text"]

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    pages = len(reader.pages)

    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content

    return text, pages

# ===== INDEXING =====

def add_to_index(text, filename, pages):
    if any(file["filename"] == filename for file in metadata):
        return

    embedding = embed_model.encode(text)

    score = 0
    if "final" in filename.lower():
        score += 2
    if "v2" in filename.lower() or "v3" in filename.lower():
        score += 1
    if len(text) > 2000:
        score += 1

    documents.append(text)

    metadata.append({
        "filename": filename,
        "importance_score": score,
        "pages": pages,
        "words": len(text.split()),
        "content": text,
        "embedding": embedding
    })

def search_query(query):
    if not metadata:
        return "No documents indexed yet."

    query_vec = embed_model.encode(query)

    best_match = None
    best_score = -1

    for file in metadata:
        doc_vec = file["embedding"]

        sim = np.dot(query_vec, doc_vec) / (
            np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
        )

        if sim > best_score:
            best_score = sim
            best_match = file["content"]

    return best_match

# ===== ANALYTICS =====

def get_dashboard_stats():
    total_files = len(metadata)
    total_pages = sum(file["pages"] for file in metadata)
    total_words = sum(file["words"] for file in metadata)
    return total_files, total_pages, total_words

def get_important_files():
    return sorted(metadata, key=lambda x: x["importance_score"], reverse=True)

def check_similarity():
    similarities = []

    for i in range(len(metadata)):
        for j in range(i + 1, len(metadata)):

            vec1 = metadata[i]["embedding"]
            vec2 = metadata[j]["embedding"]

            similarity = np.dot(vec1, vec2) / (
                np.linalg.norm(vec1) * np.linalg.norm(vec2)
            )

            similarities.append((
                metadata[i]["filename"],
                metadata[j]["filename"],
                similarity
            ))

    return similarities

# ===== MEMORY TIMELINE =====

def generate_timeline():
    timeline = []

    for file1, file2, sim in check_similarity():
        if sim > 0.70:

            doc1 = next(f for f in metadata if f["filename"] == file1)
            doc2 = next(f for f in metadata if f["filename"] == file2)

            prompt = f"""
            Compare these two document versions and explain how they evolved.

            Document A:
            {doc1["content"][:1500]}

            Document B:
            {doc2["content"][:1500]}

            Explain improvements, additions, or structural changes.
            """

            evolution = generate_response(prompt)

            timeline.append((file1, file2, evolution))

    return timeline
