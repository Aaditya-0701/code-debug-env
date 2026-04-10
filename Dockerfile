# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — CodeDebugEnv OpenEnv Environment
#
# WHY this base image?
#   ghcr.io/meta-pytorch/openenv-base already has Python 3.11, uv, and the
#   openenv-core system dependencies pre-installed. Faster builds, fewer issues.
#
# HOW HuggingFace Spaces uses this file:
#   1. When you push to the HF Space repo, HF runs `docker build .`
#   2. The container starts and runs your CMD
#   3. HF exposes port 7860 (we run uvicorn on 7860)
#   4. Judges ping https://<your-username>-code-debug-env.hf.space/health
# ─────────────────────────────────────────────────────────────────────────────

FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# ── Copy dependency manifest first (for Docker layer caching) ─────────────────
COPY pyproject.toml ./

# ── Copy the full project source ──────────────────────────────────────────────
COPY . .

# ── Install dependencies ───────────────────────────────────────────────────────
# uv is pre-installed in the openenv-base image.
# We install in --system mode so packages go to the system Python (no venv).
# Fallback to pip with server/requirements.txt if uv is not available.
RUN if command -v uv > /dev/null 2>&1; then \
        uv pip install --system -r server/requirements.txt; \
    else \
        pip install --no-cache-dir -r server/requirements.txt; \
    fi

# ── Set PYTHONPATH so imports from root (models.py etc.) resolve ──────────────
ENV PYTHONPATH="/app:$PYTHONPATH"

# ── Enable the interactive Web UI ─────────────────────────────────────────────
ENV ENABLE_WEB_INTERFACE=true

# ── Health check (judges use GET /health to confirm the Space is live) ─────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')" || exit 1

# ── HuggingFace Spaces requires port 7860 ─────────────────────────────────────
EXPOSE 7860

# ── Start the FastAPI server ───────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
