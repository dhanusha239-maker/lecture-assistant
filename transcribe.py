import whisper
import json
import os

print("Loading Whisper model...")
model = whisper.load_model("base")

audio_path = "recordings/Day 1-14 April 2026.mp4"

print("Starting transcription...")
result = model.transcribe(audio_path)

print("\nSaving structured transcript...")

segments = []

for seg in result["segments"]:
    segments.append({
        "start": seg["start"],
        "end": seg["end"],
        "text": seg["text"]
    })

os.makedirs("transcripts", exist_ok=True)

with open("transcripts/class1.json", "w", encoding="utf-8") as f:
    json.dump(segments, f, indent=2)

print("\nSaved to transcripts/class1.json")