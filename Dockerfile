# syntax=docker/dockerfile:1
# Torch flavor is selected at build time via --build-arg TORCH_FLAVOR=cpu|rocm
# (devices like /dev/kfd are NOT visible during `docker build`, so runtime GPU
# detection belongs to `docker run --device ...`, not the build).
#
# Build (CPU, default):  docker build -t video-amd .
# Build (AMD ROCm):      docker build --build-arg TORCH_FLAVOR=rocm -t video-amd .
# Run (judge-style, credentials baked via .env):
#   docker run --rm \
#     -v "$(pwd)/input:/input:ro" \
#     -v "$(pwd)/output:/output" \
#     video-amd
#
# Runtime env vars (-e FIREWORKS_API_KEY ...) override the baked .env defaults.
# The Track 2 harness injects nothing, so the image must be self-contained.
FROM ubuntu:22.04

ARG TORCH_FLAVOR=cpu

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Base OS + Python
RUN set -eux; \
    apt-get update -qq && apt-get install -y -qq --no-install-recommends \
        python3 python3-pip ca-certificates libgl1 libglib2.0-0 tini curl gnupg && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Torch install: flavor chosen explicitly at build time.
RUN set -eux; \
    if [ "$TORCH_FLAVOR" = "rocm" ]; then \
        echo "[build] TORCH_FLAVOR=rocm -> torch+rocm6.2"; \
        pip3 install -q --no-cache-dir \
            torch==2.4.1+rocm6.2 \
            torchvision==0.19.1+rocm6.2 \
            --index-url https://download.pytorch.org/whl/rocm6.2; \
    else \
        echo "[build] TORCH_FLAVOR=cpu -> torch+cpu"; \
        pip3 install -q --no-cache-dir \
            torch==2.4.1+cpu \
            torchvision==0.19.1+cpu \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    rm -rf /usr/share/doc/* /usr/share/man/* /tmp/* /root/.cache/pip

WORKDIR /app
COPY requirements-docker.txt requirements-docker-nodeps.txt ./

# Two-stage pip install:
#   1. Normal deps from requirements-docker.txt.
#   2. scenedetect + clip with --no-deps to skip heavy transitive deps
#      (av ~50 MB; torch is already installed above).
RUN set -eux; \
    pip install -q --no-cache-dir -r requirements-docker.txt && \
    pip install -q --no-cache-dir --no-deps -r requirements-docker-nodeps.txt && \
    rm -rf /root/.cache/pip /tmp/*

# Preload CLIP weights and KEEP them so the container works offline and
# doesn't re-download ~350 MB on every cold start.
RUN python -c "import clip; clip.load('ViT-B/32')"

COPY main.py config.py scene_detector.py frame_embedder.py frame_selector.py ./
COPY services/ ./services/
COPY prompts/  ./prompts/
# Track 2 harness injects NO env vars ("use your own credentials inside the
# container"), so .env ships in the image. Use disposable keys with spend caps
# and revoke them after judging. Runtime -e vars still override these defaults.
COPY .env ./
COPY docker-entrypoint.sh /usr/local/bin/

RUN mkdir -p /input /output && \
    chmod +x /usr/local/bin/docker-entrypoint.sh && \
    find /app -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
