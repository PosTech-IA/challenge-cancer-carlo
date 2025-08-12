FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set up working directory
WORKDIR /usr/src/app

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements as root first
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /usr/src/app

# Switch to non-root user
USER appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/ ./data/

CMD ["python", "src/main.py"]