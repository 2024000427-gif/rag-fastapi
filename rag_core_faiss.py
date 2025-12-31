import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

# =========================
# QUERY REWRITER (FLAN)
# =========================

rewrite_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
rewrite_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def rewrite_query(query: str) -> str:
    prompt = f"Rewrite the question for better semantic search:\n{query}"
    inputs = rewrite_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = rewrite_model.generate(inputs.input_ids, max_new_tokens=32)
    return rewrite_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# =========================
# EMBEDDINGS
# =========================

embed_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
embed_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_embedding(text: str) -> np.ndarray:
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1)
    return emb[0].numpy().astype("float32")

# =========================
# LOAD DATA
# =========================

with open("data.txt", "r", encoding="utf-8") as f:
    CHUNKS = [line.strip() for line in f if line.strip()]

embeddings = np.vstack([get_embedding(c) for c in CHUNKS])

# =========================
# FAISS INDEX
# =========================

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)   # L2 distance
index.add(embeddings)

# =========================
# GENERATION MODEL
# =========================

gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# =========================
# MAIN RAG
# =========================

def run_rag(user_query: str) -> str:
    rewritten = rewrite_query(user_query)
    query_emb = get_embedding(rewritten).reshape(1, -1)

    D, I = index.search(query_emb, k=3)

    if D[0][0] > 1.2:   # distance threshold
        return "I don't know based on the given data."

    context = "\n".join(f"- {CHUNKS[i]}" for i in I[0])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{user_query}

Answer:
"""

    inputs = gen_tokenizer(prompt, return_tensors="pt")
    outputs = gen_model.generate(inputs.input_ids, max_new_tokens=80)

    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
