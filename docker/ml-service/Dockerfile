FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .
# Install uv
RUN pip install --no-cache-dir uv
RUN uv venv
RUN uv ./venv/bin/activate

# Install Python dependencies
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/

# Create directories
RUN mkdir -p /app/models /app/notebooks /app/data

# Set environment variables
ENV PYTHONPATH=/app
ENV OUTPUT_DIR=/app/models

# Default command (can be overridden)
CMD ["uv", "run", "-m", "src.ml.train", "--num-policies", "1000", "--model-type", "decision_tree", "--output-dir", "/app/models"]
