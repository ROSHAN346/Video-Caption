# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip install --no-cache-dir "numpy<2"

# Pre-download CLIP weights into the image so the vision pipeline works
# without a runtime download (and tolerates sandboxes with no egress).
RUN python -c "import clip; clip.load('ViT-B/32')"

COPY . .

RUN mkdir -p /input /output

ENTRYPOINT ["python", "-c", "from main import competition_main; competition_main()"]
