"""
Image Analysis Service

Analyzes keyframes using Vision models and produces
structured JSON descriptions of each frame.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from services.fireworks_client import FireworksClient

logger = logging.getLogger(__name__)

# Analysis prompt template
ANALYSIS_PROMPT = """Analyze this image and return a JSON object with the following fields:
- scene_type: (indoor/outdoor/mixed/abstract)
- location: (office/street/nature/room/etc. - be specific)
- people: (count and brief description if visible, "none" if not)
- objects: (list of main objects visible)
- activities: (what is happening in the scene)
- summary: (1-2 sentence description of the scene)

Return ONLY valid JSON, no other text, no markdown code blocks."""

MULTI_IMAGE_PROMPT = """Analyze this sequence of chronological keyframes from a video and return a JSON object summarizing the ENTIRE sequence with the following fields:
- scene_type: (indoor/outdoor/mixed/abstract)
- location: (office/street/nature/room/etc. - be specific)
- people: (count and brief description if visible, "none" if not)
- objects: (list of main objects visible)
- activities: (describe the chronological sequence of actions and causal events happening across the frames)
- summary: (2-3 sentence overall summary of what happens in the video clip)

Return ONLY valid JSON, no other text, no markdown code blocks."""


def analyze_keyframe(
    image_path: str,
    client: FireworksClient,
    model: str = None,
    scene_id: int = 0,
    frame_index: int = 0
) -> dict:
    """
    Analyze a single keyframe and return structured JSON.

    Args:
        image_path: Path to the keyframe image
        client: Fireworks client instance
        model: Vision model to use (uses config default if None)
        scene_id: Scene ID for metadata
        frame_index: Frame index for metadata

    Returns:
        Dictionary with structured analysis
    """
    from config import FIREWORKS_VISION_MODEL
    default_model = FIREWORKS_VISION_MODEL
    model = model or default_model
    image_path = Path(image_path)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return _empty_analysis(scene_id, frame_index)

    try:
        # Call vision model
        logger.info(f"Calling vision model {model} for frame {frame_index}...")
        response = client.analyze_image(
            image_path=str(image_path),
            prompt=ANALYSIS_PROMPT,
            model=model,
            max_tokens=1024
        )
        logger.debug(f"Raw response: {response[:200]}...")

        # Parse JSON response
        analysis = _parse_json_response(response)

        # Add metadata
        analysis["scene_id"] = scene_id
        analysis["frame_index"] = frame_index
        analysis["image_path"] = image_path.name

        logger.info(f"Analyzed frame {frame_index} in scene {scene_id}")
        return analysis

    except Exception as e:
        logger.warning(f"API failed for frame {frame_index}: {e}. Using placeholder analysis.")
        return _empty_analysis(scene_id, frame_index)


def analyze_keyframes_batch(
    image_paths: list[str],
    client: FireworksClient,
    model: str = None,
    scene_id: int = 0
) -> list[dict]:
    """
    Analyze a batch of keyframes.

    Args:
        image_paths: List of image paths
        client: Fireworks client instance
        model: Vision model to use
        scene_id: Scene ID for metadata

    Returns:
        List of analysis dictionaries
    """
    analyses = []

    for idx, image_path in enumerate(image_paths):
        analysis = analyze_keyframe(
            image_path=image_path,
            client=client,
            model=model,
            scene_id=scene_id,
            frame_index=idx
        )
        analyses.append(analysis)

    return analyses


def analyze_keyframes_causal(
    image_paths: list[str],
    client: FireworksClient,
    model: str = None
) -> dict:
    """
    Analyze an entire sequence of keyframes in one API call to preserve causality.
    """
    from config import FIREWORKS_VISION_MODEL
    model = model or FIREWORKS_VISION_MODEL
    
    if not image_paths:
        return _empty_analysis(0, 0)
        
    try:
        logger.info(f"Calling vision model {model} for a causal sequence of {len(image_paths)} frames...")
        response = client.analyze_images_multi(
            image_paths=image_paths,
            prompt=MULTI_IMAGE_PROMPT,
            model=model,
            max_tokens=1024
        )
        
        analysis = _parse_json_response(response)
        analysis["scene_id"] = "all"
        analysis["frame_index"] = 0
        analysis["image_path"] = "multi-image-sequence"
        
        logger.info(f"Analyzed sequence successfully: {analysis.get('summary', 'N/A')[:50]}...")
        return analysis
    except Exception as e:
        logger.warning(f"API failed for multi-image sequence: {e}. Using placeholder.")
        return _empty_analysis(0, 0)


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
            "activities": "unknown",
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
        "activities": "unknown",
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
