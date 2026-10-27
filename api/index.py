"""
Vercel Serverless Entrypoint.

Wraps the full FastAPI app import in error handling so that any
startup crash is visible in Vercel logs instead of a generic 500.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` and `backend.*` resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend.main import app  # noqa: F401 — Vercel detects `app`
except Exception:
    # Fallback: surface the real traceback on the Vercel deployment URL
    import traceback
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse

    app = FastAPI()
    _tb = traceback.format_exc()

    @app.get("/{path:path}")
    def startup_error(path: str = ""):
        return PlainTextResponse(
            f"App failed to start:\n\n{_tb}", status_code=500
        )
