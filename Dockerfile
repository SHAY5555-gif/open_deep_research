FROM python:3.11-slim

ENV UV_SYSTEM_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=.venv

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY . /app

RUN uv sync --frozen --python 3.11

ENV PATH="/app/.venv/bin:${PATH}"

# Keep the container alive; Smithery startCommand will exec the LangGraph server.
CMD ["sleep", "infinity"]
