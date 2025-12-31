from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 1. Load embedding model (NOT a text generator)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# 2. Read data
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 3. Chunking (simple version)
chunks = [chunk.strip() for chunk in text.split("\n") if chunk.strip()]

# 4. Function to create embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings[0].numpy()

# 5. Create embeddings for all chunks
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# 6. User query
query = "What is a python?"
query_embedding = get_embedding(query)

# 7. Similarity search (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

scores = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

# 8. Get best match
best_index = np.argmax(scores)

print("Query:", query)
print("Best match:")
print(chunks[best_index])
