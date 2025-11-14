from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load your .env so HUGGINGFACEHUB_API_TOKEN is available
load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token is None:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set. Check your .env file.")

client = InferenceClient(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    token=token,
)

resp = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "user", "content": "Just reply with: OK"}
    ],
    max_tokens=20,
)

print("MODEL REPLY:", resp.choices[0].message["content"])
