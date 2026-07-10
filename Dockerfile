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

# Embed API credentials at build time (hackathon spec: "use your own credentials inside the container")
# Build with: docker build --build-arg FIREWORKS_API_KEY=xxx --build-arg GROQ_API_KEY=xxx .
ARG FIREWORKS_API_KEY
ARG GROQ_API_KEY
ENV FIREWORKS_API_KEY=$FIREWORKS_API_KEY
ENV GROQ_API_KEY=$GROQ_API_KEY

# Create placeholder directories for evaluation mounts
RUN mkdir -p /input /output

# Exit 0 on success, non-zero on failure (required by spec)
ENTRYPOINT ["python", "run_agent.py"]
