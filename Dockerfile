FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY ["requirements.txt", "./"]

# Install dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy application files
COPY ["predict.py", "xgb_model_eta=0.05_depth=3_min-child=6_v0.0.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["sh", "-c", "gunicorn --bind=0.0.0.0:${PORT:-9696} predict:app"]