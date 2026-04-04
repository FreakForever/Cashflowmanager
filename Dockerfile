# =========================
# Base Image
# =========================
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install git (needed for dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app/env
WORKDIR /app/env

# Ensure uv exists
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi


# =========================
# Runtime
# =========================
FROM ${BASE_IMAGE}

WORKDIR /app/env

# Copy virtual env
COPY --from=builder /app/env/.venv /app/.venv

# Copy code
COPY --from=builder /app/env /app/env

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# IMPORTANT: Gradio port
EXPOSE 7860

# Optional lightweight healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:7860 || exit 1

# Run Gradio app
CMD ["python", "server/app.py"]