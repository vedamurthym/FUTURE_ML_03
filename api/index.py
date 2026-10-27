"""
Vercel Serverless Entrypoint.
Exposes the FastAPI application instance `app`.
Vercel natively supports ASGI FastAPI applications via the `app` symbol.
"""
import sys
from pathlib import Path

# Add project root to sys.path so modules under src and backend resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app
