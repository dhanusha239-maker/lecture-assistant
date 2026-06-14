from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))