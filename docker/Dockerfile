# sky-finance application image
# Used by celery-worker, celery-beat, flower in docker-compose.app.yml

FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies (cached layer)
COPY pyproject.toml .
RUN uv sync --no-dev

# Copy application source
COPY src/ ./src/
COPY config/ ./config/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

CMD ["celery", "-A", "sky_finance.scheduler.celery_app", "worker", "--loglevel=info"]
