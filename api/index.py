"""
Vercel Serverless Entrypoint.
Imports FastAPI app and exposes ASGI handler for Vercel Python runtime.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

# Standard ASGI handler for Vercel
# Mangum converts AWS Lambda / Vercel Serverless events to ASGI requests
try:
    from mangum import Mangum
    handler = Mangum(app)
except ImportError:
    handler = app
