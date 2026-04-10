"""
FastAPI application for the CodeDebugEnv Environment.

create_app() from openenv-core automatically generates ALL the endpoints:
    GET  /health     → judges ping this to confirm the Space is live
    POST /reset      → start a new episode
    POST /step       → send an action, get an observation + reward
    GET  /state      → get current episode metadata
    GET  /docs       → interactive Swagger UI (auto-generated)
    GET  /web        → built-in browser UI for manual testing

Usage (local dev):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Usage (production / HF Space via Dockerfile):
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

import sys
import os

# Ensure root of the project is on sys.path so `from models import ...` works
# regardless of how the server is started (uvicorn, python -m, Docker)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app

from server.code_debug_env_environment import CodeDebugEnvironment
from models import CodeDebugAction, CodeDebugObservation

os.environ["ENABLE_WEB_INTERFACE"] = "true"
app = create_app(
    CodeDebugEnvironment,
    CodeDebugAction,
    CodeDebugObservation,
    env_name="code_debug_env",
)


def main():
    """Entry point for: uv run server  (defined in pyproject.toml scripts)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
