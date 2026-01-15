import time
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Embedding + memory setup
# ---------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()
memory_collection = chroma.create_collection("memory")

# ---------------------------
# Save message
# ---------------------------
def save_message(message, role="user", linked_question=None):
    embedding = embed_model.encode(message, show_progress_bar=False).tolist()
    metadata = {"role": role}
    if linked_question:
        metadata["linked_question"] = linked_question

    memory_collection.add(
        ids=[str(time.time())],
        embeddings=[embedding],
        metadatas=[metadata],
        documents=[message]
    )

# ---------------------------
# Retrieve memory
# ---------------------------
def retrieve_relevant_memory(query, k=3):
    embedding = embed_model.encode(query, show_progress_bar=False).tolist()
    result = memory_collection.query(
        query_embeddings=[embedding],
        n_results=k
    )

    docs = result.get("documents", [[]])
    if not docs or not docs[0]:
        return []

    return list(dict.fromkeys(docs[0]))

# ---------------------------
# Semantic cache
# ---------------------------
def get_memory_cached_answer(query, similarity_threshold=0.92):
    query_emb = embed_model.encode(query, show_progress_bar=False).reshape(1, -1)

    result = memory_collection.query(
        query_embeddings=[query_emb.tolist()[0]],
        n_results=5
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    if not documents or not metadatas:
        return None

    for meta, doc in zip(metadatas, documents):
        if meta.get("role") == "user":
            past_emb = embed_model.encode(doc, show_progress_bar=False).reshape(1, -1)
            sim = cosine_similarity(query_emb, past_emb)[0][0]

            if sim >= similarity_threshold:
                for meta2, doc2 in zip(metadatas, documents):
                    if meta2.get("role") == "bot" and meta2.get("linked_question") == doc:
                        return doc2
    return None
