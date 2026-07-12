"""
Image Analysis Service

Analyzes keyframes using the Fireworks vision model and produces
structured JSON descriptions of each frame.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import cv2
from services.fireworks_client import FireworksClient

logger = logging.getLogger(__name__)

# Analysis prompt template
ANALYSIS_PROMPT = """Analyze this image and return a JSON object with the following fields:
- scene_type: (indoor/outdoor/mixed/abstract)
- location: (office/street/nature/room/etc. - be specific)
- people: (count and brief description if visible, "none" if not)
- objects: (list of main objects visible)
- vehicles: (list if any, empty list if none)
- animals: (list if any, empty list if none)
- activities: (what is happening in the scene)
- weather: (sunny/cloudy/rainy/indoor-lighting/etc.)
- time_of_day: (morning/afternoon/evening/night/unknown)
- environment: (urban/rural/industrial/natural/etc.)
- risk_level: (low/medium/high with brief reason)
- confidence: (0.0-1.0 how confident you are in this analysis)
- summary: (1-2 sentence description of the scene)

Return ONLY valid JSON, no other text, no markdown code blocks."""


def analyze_keyframe_array(
    image: "cv2.typing.MatLike",
    hf_client: FireworksClient,
    model: str = None,
    scene_id: int = 0,
    frame_index: int = 0
) -> dict:
    """Analyze a keyframe held in memory (BGR numpy array) without touching disk.

    Encodes the frame to JPEG bytes and calls the vision model directly, skipping
    the write-JPEG-then-re-read round trip used by analyze_keyframe(image_path=...).
    """
    from config import FIREWORKS_VISION_MODEL
    model = model or FIREWORKS_VISION_MODEL

    try:
        ok, buf = cv2.imencode(".jpg", image)
        if not ok:
            logger.error(f"Failed to encode frame {frame_index}")
            return _empty_analysis(scene_id, frame_index)

        logger.info(f"Calling vision model {model} for frame {frame_index} (in-memory)...")
        response = hf_client.analyze_image_base64(
            image_bytes=buf.tobytes(),
            prompt=ANALYSIS_PROMPT,
            model=model,
            max_tokens=1024
        )

        analysis = _parse_json_response(response)
        analysis["scene_id"] = scene_id
        analysis["frame_index"] = frame_index
        analysis["image_path"] = ""
        logger.info(f"Analyzed frame {frame_index} in scene {scene_id}")
        return analysis
    except Exception as e:
        logger.warning(f"API failed for frame {frame_index}: {e}. Using placeholder analysis.")
        return _empty_analysis(scene_id, frame_index)


def _parse_json_response(response: str) -> dict:
    """
    Parse JSON from model response, handling common issues.

    Args:
        response: Raw model response text

    Returns:
        Parsed dictionary
    """
    # Remove markdown code blocks if present
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Strip whitespace
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        logger.debug(f"Raw response: {response[:500]}")

        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return empty analysis on parse failure
        return {
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
            "risk_level": "unknown",
            "confidence": 0.0,
            "summary": "Analysis failed to parse"
        }


def _empty_analysis(scene_id: int, frame_index: int) -> dict:
    """Return empty analysis structure."""
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
        "risk_level": "unknown",
        "confidence": 0.0,
        "summary": "Analysis unavailable"
    }


def save_analysis(analysis: dict, output_path: str):
    """Save analysis to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(f"Saved analysis to: {output_path}")


def load_analysis(json_path: str) -> dict:
    """Load analysis from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)
