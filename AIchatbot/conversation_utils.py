import yaml
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CACHE_FILE = "conversation_cache.yaml"
SIMILARITY_THRESHOLD = 0.8

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Ensure cache file exists
# ---------------------------
def init_cache():
    if not os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump({"conversations": []}, f)

init_cache()

# ---------------------------
# Load cache 
# ---------------------------
def load_cache():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {"conversations": []}

# ---------------------------
# Save cache
# ---------------------------
def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(cache, f, allow_unicode=True, sort_keys=False)

# ---------------------------
# Get cached answer (semantic)
# ---------------------------
def get_cached_answer(user_question):
    cache = load_cache()
    if not cache["conversations"]:
        return None

    user_emb = embed_model.encode(user_question, show_progress_bar=False).reshape(1, -1)

    questions = [c["question"] for c in cache["conversations"]]
    answers = [c["answer"] for c in cache["conversations"]]

    q_embs = embed_model.encode(questions, show_progress_bar=False)
    sims = cosine_similarity(user_emb, q_embs)[0]

    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]

    if best_score >= SIMILARITY_THRESHOLD:
        return answers[best_idx]

    return None

# ---------------------------
# Add to cache 
# ---------------------------
def add_to_cache(user_question, bot_answer):
    cache = load_cache()

    cache["conversations"].append({
        "question": user_question,
        "answer": bot_answer
    })

    save_cache(cache)

# ---------------------------
# FAQ semantic match
# ---------------------------
def get_faq_match(query, questions, embeddings, threshold=0.65):
    query_emb = embed_model.encode(query, show_progress_bar=False).reshape(1, -1)
    
    embeddings_array = np.vstack(embeddings)
    
    sims = cosine_similarity(query_emb, embeddings_array)[0]
    
    best_index = int(np.argmax(sims))
    best_score = sims[best_index]
    
    if best_score >= threshold:
        return best_index
    
    return None
