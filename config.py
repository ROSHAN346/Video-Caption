import os
from dotenv import load_dotenv

load_dotenv()

DETECTOR_CONFIG = {
    "adaptive_threshold": 3.0,
    "min_scene_len": 15,
    "window_width": 2,
    "min_content_val": 15.0,
}

# Default frame sampling strategy for legacy single-frame per scene sampler
# (frame_sampler.extract_frames).
FRAME_STRATEGY = "middle"

# --- Embedding-based keyframe selection config (this phase) ---
# Hard ceiling on total selected keyframes for ANY video (single- or multi-scene).
MAX_FRAMES = 5
# Dense candidate sampling rate inside each scene (frames per second).
# Higher = finer temporal coverage (smaller chance of missing a brief event),
# at the cost of more embedding work. Bumped from 2.5 -> 5.0 to shrink the
# fixed-rate sampling blind spot for short/rare occurrences.
CANDIDATE_FPS = 2
# CLIP model variant. ViT-B/32 is the lightest reasonable option for CPU inference.
CLIP_MODEL_NAME = "ViT-B/32"
# Frames embedded per forward pass (kept modest for 16GB / no-GPU machines).
EMBEDDING_BATCH_SIZE = 32
# Min max-min embedding distance required to keep adding keyframes. When the next
# best candidate is closer than this (i.e. near-duplicate of already-selected
# frames), selection stops early. 0 disables early-stop (always fill MAX_FRAMES).
# Additive to MAX_FRAMES: never exceeds the cap, only stops sooner on redundant
# content (e.g. a static scene where only 1-2 frames are meaningful).
EARLY_STOP_MIN_DIST = 0.03

# --- Hugging Face Configuration ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_VISION_MODEL = "MiniMaxAI/MiniMax-M3"
HF_TEXT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_VISION_PROVIDER = "novita"
HF_TEXT_PROVIDER = "novita"

# --- Fireworks AI Configuration ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
FIREWORKS_VISION_MODEL = os.getenv("FIREWORKS_VISION_MODEL", "accounts/fireworks/models/minimax-m3")
FIREWORKS_TEXT_MODEL = os.getenv("FIREWORKS_TEXT_MODEL", "accounts/fireworks/models/qwen3-8b")

# --- Provider Selection (huggingface / fireworks) ---
AI_PROVIDER = os.getenv("AI_PROVIDER", "huggingface")

# --- Report Generation ---
REPORT_STYLES = [
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
    "jargon"
]
DEFAULT_REPORT_STYLE = "formal"
REPORT_CACHE_ENABLED = True
REPORT_LANGUAGE = "en"

# --- Scene JSON Schema ---
SCENE_JSON_FIELDS = [
    "scene_id", "scene_type", "location", "people", "objects",
    "vehicles", "animals", "activities", "weather", "time_of_day",
    "environment", "risk_level", "confidence", "summary"
]
