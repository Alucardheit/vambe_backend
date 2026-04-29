FROM python:3.12-slim-bookworm

# Install system dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        build-essential; \
    apt-get purge -y --auto-remove; \
    rm -rf /var/lib/apt/lists/*
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set workdir
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies with uv
RUN uv sync

# Expose the port
EXPOSE 8000


CMD uv run uvicorn src.main:app --host 0.0.0.0 --log-level trace