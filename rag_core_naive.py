from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import numpy as np

# =========================
# QUERY REWRITER (FLAN)
# =========================

rewrite_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
rewrite_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def rewrite_query(query: str) -> str:
    prompt = (
        "Rewrite the following question to be clear and optimized for search.\n\n"
        f"Question: {query}\n\nRewritten:"
    )
    inputs = rewrite_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = rewrite_model.generate(inputs.input_ids, max_new_tokens=32)
    return rewrite_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# =========================
# EMBEDDINGS (MiniLM)
# =========================

embed_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
embed_model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_embedding(text: str):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].numpy()


# =========================
# LOAD DATA ONCE
# =========================

with open("data.txt", "r", encoding="utf-8") as f:
    CHUNKS = [line.strip() for line in f if line.strip()]

CHUNK_EMBEDDINGS = [get_embedding(c) for c in CHUNKS]


# =========================
# SIMILARITY
# =========================

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================
# GENERATION MODEL (FLAN)
# =========================

gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


# =========================
# MAIN RAG FUNCTION
# =========================

def run_rag(user_query: str) -> str:
    rewritten_query = rewrite_query(user_query)

    query_embedding = get_embedding(rewritten_query)
    scores = [cosine_similarity(query_embedding, emb) for emb in CHUNK_EMBEDDINGS]

    best_score = max(scores)
    if best_score < 0.45:
        return "I don't know based on the provided context."

    top_k = 3
    top_indices = np.argsort(scores)[-top_k:][::-1]

    context_blocks = []
    for idx, i in enumerate(top_indices, start=1):
        context_blocks.append(f"[{idx}] {CHUNKS[i]}")

    context: str = "\n".join(context_blocks)


    prompt = (
    "Answer the question using ONLY the numbered context below.\n"
    "Cite the source number(s) in your answer like [1], [2].\n"
    "If the answer is not in the context, say: I don't know.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {user_query}\n\nAnswer:"
)


    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = gen_model.generate(inputs.input_ids, max_new_tokens=80)

    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
