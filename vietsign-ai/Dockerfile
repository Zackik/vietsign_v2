FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
CMD ["python", "-m", "vietsign.cli", "serve", "--config", "configs/api/default.yaml"]
