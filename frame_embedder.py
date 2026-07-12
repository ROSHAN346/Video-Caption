import threading

import torch
import clip
import numpy as np
from PIL import Image

from config import CLIP_MODEL_NAME, EMBEDDING_BATCH_SIZE

# Auto-detect device: CUDA (NVIDIA or ROCm/AMD) -> DirectML (AMD/Intel on Windows)
# if installed -> CPU. Returns a torch.device so DirectML (which exposes a
# device object rather than a string) works transparently downstream.
_model = None
_preprocess = None
# Tasks are processed by multiple threads; model load and inference are
# serialized so concurrent tasks can't race the lazy init or contend on the
# same model weights mid-forward-pass.
_model_lock = threading.Lock()


def _resolve_device():
    """Return the best available torch.device: CUDA, else DirectML, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def _get_device_info():
    """Return a human-readable device description for logging."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return f"GPU: {gpu_name} ({gpu_mem:.1f} GB)"
    try:
        import torch_directml
        return f"DirectML: {torch_directml.device()}"
    except Exception:
        return "CPU"


def load_model():
    """Load (and cache) the CLIP model + preprocess on best available device."""
    global _model, _preprocess
    with _model_lock:
        if _model is None:
            device = _resolve_device()
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
    device = _resolve_device()

    if not frames:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = []
    with _model_lock, torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tensors = []
            for f in batch:
                rgb = f[:, :, ::-1]  # BGR -> RGB (faster than cv2.cvtColor)
                tensors.append(preprocess(Image.fromarray(rgb)))
            batch_t = torch.stack(tensors).to(device)
            feats = model.encode_image(batch_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
