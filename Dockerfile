FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml playbook.json ./
COPY app/ app/
RUN pip install --no-cache-dir -e .

FROM base AS test
RUN pip install --no-cache-dir -e ".[dev]"
COPY tests/ tests/
CMD ["pytest"]

FROM base
COPY --from=web /web/dist web/dist
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
