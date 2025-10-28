# Build stage
FROM python:3.9-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY ./src ./src

ENV PATH="/opt/venv/bin/:$PATH"

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]