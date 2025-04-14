FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir uv
RUN uv venv
RUN uv ./venv/bin/activate
# Install Python dependencies and Jupyter extensions
RUN uv pip install --no-cache-dir -r requirements.txt && \
    uv pip install --no-cache-dir \
    jupyterlab \
    jupyter-contrib-nbextensions \
    jupyter-nbextensions-configurator \
    ipywidgets \
    && jupyter contrib nbextension install --user \
    && jupyter nbextensions_configurator enable --user

# Copy source code
COPY src/ /app/src/

# Create directories
RUN mkdir -p /app/models /app/notebooks /app/data

# Set environment variables
ENV PYTHONPATH=/app

# Expose Jupyter port
EXPOSE 8888

# Run Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
