from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

print("Key starts with:", key[:15])
print("Key ends with:", key[-6:])