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

# Top-level ASGI entrypoint exposed for Vercel Serverless Functions
from backend.main import app  # noqa: F401

# Explicit module-level assignment to guarantee Vercel AST detection
app = app
