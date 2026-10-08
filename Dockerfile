FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY app ./app
COPY config ./config

COPY scripts/bootstrap_model.py ./scripts/bootstrap_model.py

RUN python scripts/bootstrap_model.py

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:/app/src"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]