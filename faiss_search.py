import json
import numpy as np
import faiss

from dotenv import load_dotenv
import os

load_dotenv()

from config import client

# Load FAISS index + metadata
index = faiss.read_index("lectures.index")

with open("metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(res.data[0].embedding).astype("float32")


def search(query, top_k=8):
    query_emb = get_embedding(query).reshape(1, -1)

    distances, indices = index.search(query_emb, top_k)

    results = []

    for idx in indices[0]:
        results.append({
            "text": data[idx]["text"],
            "file": data[idx]["file"],
            "start": data[idx]["start"],
            "score": float(distances[0][list(indices[0]).index(idx)])
        })

    return results