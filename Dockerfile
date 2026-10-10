# syntax=docker/dockerfile:1
# Multi-stage build. Runtime image has no build tools; runs as non-root.
FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build \
 && python -m build --wheel --outdir /dist

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 selflearn \
 && mkdir -p /app/data && chown selflearn:selflearn /app/data
WORKDIR /app
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir "/tmp/"*.whl "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2.6" \
 && rm -f /tmp/*.whl
USER selflearn
EXPOSE 8000
ENV SELFLEARN_DB_PATH=/app/data/selflearn.db
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"
CMD ["uvicorn", "selflearn.server:app", "--host", "0.0.0.0", "--port", "8000"]
