import cv2
import torch
import clip
import numpy as np
from PIL import Image

from config import CLIP_MODEL_NAME, EMBEDDING_BATCH_SIZE

# CLIP runs on CPU only in this pipeline (no GPU assumed). Loading is cached so
# the model/weights are fetched from disk once per process.
_DEVICE = "cpu"
_model = None
_preprocess = None


def load_model():
    """Load (and cache) the CLIP model + preprocess on CPU."""
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load(CLIP_MODEL_NAME, device=_DEVICE)
        _model.eval()
    return _model, _preprocess


def embed_frames(frames: list, batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """Embed a list of BGR numpy frames -> L2-normalized embeddings array (N, D).

    Frames come from cv2 (BGR), so we convert to RGB before preprocessing.
    Embeddings are normalized to unit length so cosine similarity == dot product,
    which keeps the novelty score and farthest-point selection cheap and stable.
    """
    load_model()
    if not frames:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            tensors = []
            for f in batch:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                tensors.append(_preprocess(Image.fromarray(rgb)))
            batch_t = torch.stack(tensors).to(_DEVICE)
            feats = _model.encode_image(batch_t)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

    return np.concatenate(embeddings, axis=0)
