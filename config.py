import os
from dotenv import load_dotenv

load_dotenv()

DETECTOR_CONFIG = {
    "adaptive_threshold": 3.0,
    "min_scene_len": 15,
    "window_width": 2,
    "min_content_val": 15.0,
}

# --- Embedding-based keyframe selection config ---
# Hard ceiling on total selected keyframes for ANY video (single- or multi-scene).
# Kept low to cut embedding work and the number of vision API calls.
MAX_FRAMES = 5
# Dense candidate sampling rate inside each scene (frames per second).
# Lower = fewer candidate frames to embed (the dominant CPU cost), at the
# expense of slightly coarser temporal coverage. Set to 0.5 for aggressive
# CPU speed; coverage remains adequate for most content.
CANDIDATE_FPS = 0.5
# CLIP model variant. ViT-B/32 is the lightest reasonable option; the device
# (CUDA / DirectML / CPU) is auto-selected at runtime in frame_embedder.py.
CLIP_MODEL_NAME = "ViT-B/32"
# Frames embedded per forward pass (kept modest for 16GB / no-GPU machines).
EMBEDDING_BATCH_SIZE = 16
# Min max-min embedding distance required to keep adding keyframes. When the next
# best candidate is closer than this (i.e. near-duplicate of already-selected
# frames), selection stops early. 0 disables early-stop (always fill MAX_FRAMES).
# Additive to MAX_FRAMES: never exceeds the cap, only stops sooner on redundant
# content (e.g. a static scene where only 1-2 frames are meaningful).
EARLY_STOP_MIN_DIST = 0.0075
# Captured frames (kept in memory during the single decode pass) are downscaled
# so their longest side never exceeds this. CLIP embeds at 224px and vision-API
# JPEGs are capped at 1280px anyway, so nothing downstream loses quality while
# RAM usage on UHD sources drops ~4-10x.
MAX_CAPTURE_SIDE = 1280

# --- Video download limits (competition/batch mode) ---
# Hard cap on a single downloaded video to protect the disk.
MAX_DOWNLOAD_MB = 1024

# --- Competition runtime budget ---
# Number of tasks processed concurrently (download/decode of one task overlaps
# the API calls of another). Keep modest for small judge VMs.
MAX_TASK_WORKERS = int(os.getenv("MAX_TASK_WORKERS", "3"))
# Global wall-clock budget: the judge kills containers at 10 minutes, so we
# stop waiting at this point and write results with whatever has completed
# (unfinished tasks keep their placeholder captions — partial beats TIMEOUT).
DEADLINE_SECONDS = float(os.getenv("DEADLINE_SECONDS", "510"))

# --- Fireworks AI Configuration (vision) ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_VISION_MODEL = os.getenv("FIREWORKS_VISION_MODEL", "accounts/fireworks/models/minimax-m3")

# --- Fireworks AI Configuration (text) ---
# Separate key so vision and text can run on different Fireworks accounts.
# Falls back to the vision key when unset.
FIREWORKS_TEXT_API_KEY = os.getenv("FIREWORKS_TEXT_API_KEY", "") or FIREWORKS_API_KEY
FIREWORKS_TEXT_BASE_URL = os.getenv("FIREWORKS_TEXT_BASE_URL", FIREWORKS_BASE_URL)
FIREWORKS_TEXT_MODEL = os.getenv("FIREWORKS_TEXT_MODEL", "accounts/fireworks/models/gpt-oss-120b")

# --- Report Generation ---
REPORT_STYLES = [
    "formal",
]
DEFAULT_REPORT_STYLE = "formal"
REPORT_CACHE_ENABLED = True
REPORT_LANGUAGE = "en"

# --- Scene JSON Schema ---

