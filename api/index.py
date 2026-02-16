# Vercel serverless function: import the FastAPI app so we can set maxDuration in vercel.json.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import app
