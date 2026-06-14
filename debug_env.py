import os
from dotenv import load_dotenv

load_dotenv()

print("ENV FILE KEY:", os.getenv("OPENAI_API_KEY"))
print("SYSTEM KEY:", os.environ.get("OPENAI_API_KEY"))