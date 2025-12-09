import os
from google import genai

# Simple helper to list available models for your API key
# Usage (PowerShell):
#   $env:GEMINI_API_KEY="your_api_key"
#   python list_models.py

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

client = genai.Client(api_key=api_key)

models = client.models.list()
models_iter = getattr(models, 'models', models)
for m in models_iter:
    print(getattr(m, 'name', None) or getattr(m, 'model', None) or str(m))
