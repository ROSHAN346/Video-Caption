import cv2
import torch
import clip
import numpy as np
from PIL import Image

from config import CLIP_MODEL_NAME, EMBEDDING_BATCH_SIZE

# Auto-detect device: CUDA if available, else CPU
_model = None
_preprocess = None


def _get_device():
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_device_info():
    """Get detailed device information for console output."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"
    return "CPU"


def load_model():
    """Load (and cache) the CLIP model + preprocess on best available device."""
    global _model, _preprocess
    if _model is None:
        device = _get_device()
        device_info = _get_device_info()
        print(f"[CLIP] Device: {device_info}")
        print(f"[CLIP] Loading {CLIP_MODEL_NAME} on {device}...")
        _model, _preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        _model.eval()
        print(f"[CLIP] Model loaded successfully on {device}")
    return _model, _preprocess


def embed_frames(frames: list, batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """Embed a list of BGR numpy frames -> L2-normalized embeddings array (N, D).

    Frames come from cv2 (BGR), so we convert to RGB before preprocessing.
    Embeddings are normalized to unit length so cosine similarity == dot product,
    which keeps the novelty score and farthest-point selection cheap and stable.
    """
    model, preprocess = load_model()
    device = _get_device()

    if not frames:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tensors = []
            for f in batch:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(preprocess(Image.fromarray(rgb)))
            batch_t = torch.stack(tensors).to(device)
            feats = model.encode_image(batch_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
