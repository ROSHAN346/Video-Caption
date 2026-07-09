"""
Prompt Loader Service

Loads and formats writing-style prompt templates from the prompts/ directory.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def load_prompt(style: str) -> str:
    """
    Load a prompt template by style name.

    Args:
        style: Style name (formal, sarcastic, etc.)

    Returns:
        Prompt template string

    Raises:
        FileNotFoundError: If prompt file not found
    """
    prompt_file = PROMPTS_DIR / f"{style}.txt"

    if not prompt_file.exists():
        available = get_available_styles()
        raise FileNotFoundError(
            f"Prompt not found: {style}. "
            f"Available styles: {available}"
        )

    with open(prompt_file, "r", encoding="utf-8") as f:
        template = f.read()

    logger.debug(f"Loaded prompt template: {style}")
    return template


def format_prompt(template: str, scene_data: dict) -> str:
    """
    Insert scene data into a prompt template.

    Args:
        template: Prompt template with {scene_data} placeholder
        scene_data: Dictionary of scene analysis data

    Returns:
        Formatted prompt string
    """
    # Format scene data as readable text
    scene_text = _format_scene_data(scene_data)

    # Replace placeholder
    formatted = template.replace("{scene_data}", scene_text)

    return formatted


def get_available_styles() -> list[str]:
    """
    Get list of available prompt styles.

    Returns:
        List of style names
    """
    if not PROMPTS_DIR.exists():
        return []

    styles = []
    for prompt_file in PROMPTS_DIR.glob("*.txt"):
        styles.append(prompt_file.stem)

    return sorted(styles)


def validate_style(style: str) -> bool:
    """
    Check if a style is available.

    Args:
        style: Style name to validate

    Returns:
        True if style exists
    """
    return style in get_available_styles()


def _format_scene_data(scene_data: dict) -> str:
    """
    Format scene data dictionary as readable text.

    Args:
        scene_data: Scene analysis dictionary

    Returns:
        Formatted text
    """
    lines = []

    for key, value in scene_data.items():
        if key in ["scene_id", "frame_index", "image_path", "frame_count"]:
            continue  # Skip metadata fields

        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) if value else "none"

        lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)
