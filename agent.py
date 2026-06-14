
from faiss_search import search
from dotenv import load_dotenv
import os

load_dotenv()

from config import client

def build_context(results):
    return "\n\n".join(
        [
            f"[Source: {r.get('file','unknown')} | {r.get('start','?')} sec]\n{r['text']}"
            for r in results
        ]
    )


def ask_agent(question):
    results = search(question, top_k=8)

    context = build_context(results)

    messages = [
        {
            "role": "system",
            "content": (
            "You are a helpful AI tutor for lecture notes.\n"
            "Use ONLY given context.\n"
            "If missing, say not clearly mentioned in lectures.\n"
            "Always mention source file and timestamp if available.\n"
            "Treat each context block separately."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        print(ask_agent(q))