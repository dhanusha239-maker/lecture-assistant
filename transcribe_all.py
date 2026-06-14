import whisper
import os
import json

print("Loading model...")
model = whisper.load_model("base")

input_folder = "recordings"
output_folder = "transcripts"

os.makedirs(output_folder, exist_ok=True)

files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

print(f"Found {len(files)} videos")

for file in files:
    input_path = os.path.join(input_folder, file)

    output_name = file.replace(".mp4", ".json")
    output_path = os.path.join(output_folder, output_name)

    # Skip if already processed
    if os.path.exists(output_path):
        print(f"Skipping (already done): {file}")
        continue

    print(f"\nProcessing: {file}")

    result = model.transcribe(input_path)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"]
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2)

    print(f"Saved: {output_path}")

print("\nAll videos processed!")