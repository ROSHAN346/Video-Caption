FROM python:3.11-slim

# Install system dependencies needed for OpenCV (headless support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY run_agent.py .

# Bake credentials as a plain file (NOT ENV/ARG) so they aren't visible via `docker inspect`
COPY secrets.env /app/secrets.env
RUN chmod 600 /app/secrets.env

# Create placeholder directories for evaluation mounts
RUN mkdir -p /input /output

# Exit 0 on success, non-zero on failure (required by spec)
ENTRYPOINT ["python", "run_agent.py"]
