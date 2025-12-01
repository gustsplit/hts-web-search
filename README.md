# HTS Web Search PoC

This project is a Proof of Concept (PoC) web service that helps users:
- Search and explain US HTS codes
- Estimate duties/tariffs based on selected HTS
- Generate an email draft summarizing the selected HTS code and calculation result

## Stack

- Backend: FastAPI (Python)
- Frontend: HTML + JavaScript (Toss Bank style UI)
- Target: US HTS code classification and email reply automation

## Branch Strategy

- `main`: stable, production-ready code
- `dev`: active development (new features, changes)

## Enabling Codex / Generative AI workspace (Quick Start)

This project uses Google's Generative AI (Gemini) to provide HTS classification and related features.

Follow these steps to enable the workspace locally:

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Add your Google API key (do not commit this to source control):

Create a `.env` file at `backend/.env` with the following content:

```
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

Or run the helper script (Windows) in the repo root:

```cmd
tools\enable_codex.bat YOUR_GOOGLE_API_KEY
```

3. Verify available models:

```bash
python list_models.py --api-key YOUR_GOOGLE_API_KEY
```

4. Start the backend:

```bash
cd backend
uvicorn main:app --reload
```

Notes:
- The repository should not contain API keys. There's a placeholder `backend/.env.example` you can copy to `backend/.env` and edit.
- If you find any exposed keys, rotate them immediately and remove the secrets from the repo.

## API Endpoints (Examples)

- GET `/api/exchange-rate/usd-krw`: Returns the USD-KRW exchange rate, `source` (live-update/cache/fallback-*), and `timestamp` in ISO 8601 UTC (e.g., 2025-11-28T10:00:00Z). A deprecated alias `/api/usd-krw` also exists for backwards compatibility but will include a `deprecated` flag.


