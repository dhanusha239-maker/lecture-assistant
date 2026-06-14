import os
import json
import time
from openai import OpenAI

client = OpenAI()

TRANSCRIPT_DIR = "transcripts"
OUTPUT_FILE = "embeddings.json"


def load_transcripts():
    data = []

    for file in os.listdir(TRANSCRIPT_DIR):
        if file.endswith(".json"):
            path = os.path.join(TRANSCRIPT_DIR, file)

            with open(path, "r", encoding="utf-8") as f:
                segments = json.load(f)

            current_text = ""
            start_time = float(segments[0]["start"])

            for seg in segments:
                current_text += " " + seg["text"]

                if len(current_text) > 1000:
                    data.append({
                        "file": file,
                        "text": current_text.strip(),
                        "start": start_time
                    })

                    current_text = ""
                    start_time = float(seg["start"])

            if current_text:
                data.append({
                    "file": file,
                    "text": current_text.strip(),
                    "start": start_time
                })

    return data


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def main():
    transcripts = load_transcripts()
    embeddings = []

    print(f"Total segments: {len(transcripts)}")
    print("Generating embeddings...")

    for i, item in enumerate(transcripts):
        try:
            emb = get_embedding(item["text"])

            time.sleep(0.1)

            embeddings.append({
                "file": item["file"],
                "text": item["text"],
                "start": item["start"],
                "embedding": emb
            })

            if i % 20 == 0:
                print(f"Processed {i}/{len(transcripts)}")

        except Exception as e:
            print("Error:", e)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print("Saved embeddings to", OUTPUT_FILE)


if __name__ == "__main__":
    main()