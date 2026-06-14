import json
import numpy as np
import faiss

# Load embeddings
with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total embeddings:", len(data))

# Convert to numpy array
embeddings = np.array([item["embedding"] for item in data]).astype("float32")

dimension = embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, "lectures.index")

# Save metadata
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

print("FAISS index built successfully!")