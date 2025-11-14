# Prompt Injection Detector

## Environment Setup

1. Install dependencies (ideally in a virtual environment): `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and add your secrets:
   ```bash
   cp .env.example .env
   ```
   - `LLAMA_API_KEY`: key for your Llama API access.
   - `HUGGINGFACE_TOKEN`: token for downloading gated Hugging Face models, if required.
3. Keep `.env` out of version control (already handled in `.gitignore`).

These environment variables let you authenticate with external APIs without hard-coding secrets in the source.
