import json
import numpy as np
from openai import OpenAI

client = OpenAI()

EMBEDDINGS_FILE = "embeddings.json"


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_query_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def load_embeddings():
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def search(query, top_k=5):
    print("Creating query embedding...")
    query_embedding = get_query_embedding(query)

    data = load_embeddings()

    scored_results = []

    for item in data:
        score = cosine_similarity(query_embedding, item["embedding"])

        scored_results.append({
            "score": score,
            "file": item["file"],
            "text": item["text"],
            "start": item["start"]
        })

    scored_results.sort(key=lambda x: x["score"], reverse=True)

    return scored_results[:top_k]


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def main():
    query = input("Ask your lecture question: ")

    results = search(query)

    print("\n===== TOP RESULTS =====\n")

    for r in results:
        print(f"Lecture: {r['file']}")
        print(f"Time: {format_time(r['start'])}")
        print(f"Score: {r['score']:.3f}")
        print(f"Text: {r['text']}")
        print("-" * 50)


if __name__ == "__main__":
    main()