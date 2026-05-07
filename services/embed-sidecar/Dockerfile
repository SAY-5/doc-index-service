FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build deps for any C extensions, then drop them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app

# Default install is the lean path (HashEmbedder only). Pass
# --build-arg INSTALL_EXTRAS=hf to bake in sentence-transformers.
ARG INSTALL_EXTRAS=""
RUN if [ -n "$INSTALL_EXTRAS" ]; then \
        pip install ".[${INSTALL_EXTRAS}]"; \
    else \
        pip install . ; \
    fi \
    && apt-get purge -y --auto-remove build-essential

ENV EMBEDDER_KIND=hash
EXPOSE 8088
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8088"]
