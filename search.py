import json
import numpy as np
from openai import OpenAI

client = OpenAI()

with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_query_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding


def search(query, top_k=5):
    query_emb = get_query_embedding(query)

    scored = []

    for item in data:
        score = cosine_similarity(query_emb, item["embedding"])

        scored.append({
            "score": float(score),
            "text": item["text"],
            "file": item["file"],
            "start": item["start"]
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored[:top_k]