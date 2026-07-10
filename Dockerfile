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
COPY app.py .

# Create placeholder directories for evaluation mounts
RUN mkdir -p /input /output

# Run the agent script by default on container startup
ENTRYPOINT ["python", "run_agent.py"]
