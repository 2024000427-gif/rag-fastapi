import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# -------------------------
# Embedding model
# -------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)
model = AutoModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].numpy()

# -------------------------
# Load data
# -------------------------

with open("data.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

embeddings = np.array([get_embedding(c) for c in chunks]).astype("float32")

# -------------------------
# FAISS index
# -------------------------

dim = embeddings.shape[1]          # vector size
index = faiss.IndexFlatL2(dim)     # basic FAISS index
index.add(embeddings)              # store vectors

# -------------------------
# Query
# -------------------------

query = "What is attention in transformers?"
query_vector = np.array([get_embedding(query)]).astype("float32")

# Search
k = 2
distances, indices = index.search(query_vector, k)

print("Query:", query)
print("Top results:")
for i in indices[0]:
    print("-", chunks[i])
