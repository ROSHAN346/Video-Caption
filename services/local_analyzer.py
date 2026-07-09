"""Local image analysis using HuggingFace captioning model."""

from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)

_captioner = None


def _get_captioner():
    """Lazy load the captioning model."""
    global _captioner
    if _captioner is None:
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings('ignore')
            _captioner = pipeline('image-to-text', model='nlpconnect/vit-gpt2-image-captioning', device=-1)
            logger.info("Local captioning model loaded")
        except Exception as e:
            logger.error(f"Failed to load captioning model: {e}")
            return None
    return _captioner


def analyze_image_local(image_path: str, scene_id: int = 0, frame_index: int = 0) -> dict:
    """Analyze image using local captioning model."""
    if not os.path.exists(image_path):
        return _empty_analysis(scene_id, frame_index)

    try:
        img = Image.open(image_path)

        # Get caption from model
        captioner = _get_captioner()
        if captioner:
            result = captioner(img, max_new_tokens=50)
            caption = result[0]['generated_text'].strip()
        else:
            caption = _basic_analysis(img)

        # Get basic image properties
        width, height = img.size
        pixels = list(img.getdata())
        total = len(pixels)
        avg_r = sum(p[0] for p in pixels) / total
        avg_g = sum(p[1] for p in pixels) / total
        avg_b = sum(p[2] for p in pixels) / total
        brightness = (avg_r + avg_g + avg_b) / 3

        # Determine properties from caption and image
        time_of_day = "daytime" if brightness > 150 else "nighttime or indoor"
        scene_type = "outdoor" if any(w in caption.lower() for w in ["street", "road", "building", "sky", "car", "tree", "park"]) else "indoor"

        return {
            "scene_id": scene_id,
            "frame_index": frame_index,
            "scene_type": scene_type,
            "location": caption.split(" with ")[0] if " with " in caption else caption[:50],
            "people": "visible" if any(w in caption.lower() for w in ["person", "people", "man", "woman", "pedestrian"]) else "none visible",
            "objects": [w for w in caption.split() if w in ["car", "cars", "building", "buildings", "tree", "trees", "sign", "bus", "truck", "bicycle", "bike"]],
            "vehicles": [w for w in ["car", "cars", "bus", "truck", "bicycle"] if w in caption.lower()],
            "animals": [],
            "activities": caption,
            "weather": "unknown",
            "time_of_day": time_of_day,
            "environment": "urban" if any(w in caption.lower() for w in ["street", "road", "building", "city"]) else "other",
            "risk_level": "low",
            "confidence": 0.7,
            "summary": f"Scene shows: {caption}"
        }
    except Exception as e:
        logger.error(f"Local analysis failed: {e}")
        return _empty_analysis(scene_id, frame_index)


def _basic_analysis(img: Image.Image) -> str:
    """Fallback basic analysis if model fails."""
    width, height = img.size
    aspect = "landscape" if width > height else "portrait" if height > width else "square"
    return f"a {aspect} image with visual content"


def _empty_analysis(scene_id: int, frame_index: int) -> dict:
    return {
        "scene_id": scene_id,
        "frame_index": frame_index,
        "scene_type": "unknown",
        "location": "unknown",
        "people": "unknown",
        "objects": [],
        "vehicles": [],
        "animals": [],
        "activities": "unknown",
        "weather": "unknown",
        "time_of_day": "unknown",
        "environment": "unknown",
        "risk_level": "low",
        "confidence": 0.0,
        "summary": "Analysis unavailable"
    }
