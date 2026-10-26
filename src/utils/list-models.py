# integrations
import os
from dotenv import load_dotenv; load_dotenv()

from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# list available embedding models
for model in client.models.list():
    print(f"Model ID: {model.name}")
    print(f"Supported Operations: {model.supported_actions}\n")