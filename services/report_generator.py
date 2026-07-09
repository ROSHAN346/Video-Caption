"""
Report Generator Service

Generates multi-tone reports using Hugging Face text models.
"""

import logging
from typing import Optional

from services.hf_client import HFClient
from services.prompt_loader import load_prompt, format_prompt

logger = logging.getLogger(__name__)

# System prompts for each style
SYSTEM_PROMPTS = {
    "formal": "You are a professional report writer. Summarize all activities from the video into exactly 2 short factual sentences. No headers, no bullet points.",
    "sarcastic": "You are a witty, sarcastic commentator. Summarize all activities from the video into exactly 2 short sarcastic sentences. No headers, no bullet points.",
    "humorous_tech": "You are a programmer who finds tech hilarious. Summarize all activities from the video into exactly 2 short funny sentences with programming references.",
    "humorous_non_tech": "You are a stand-up comedian. Summarize all activities from the video into exactly 2 short funny sentences.",
    "jargon": "You are a domain expert. Summarize all activities from the video into exactly 2 short sentences heavy with professional jargon."
}


def generate_report(
    scene_data: dict,
    style: str,
    hf_client: HFClient,
    model: str = None
) -> str:
    """
    Generate a single report in the specified style.

    Args:
        scene_data: Aggregated scene analysis
        style: Writing style (formal, sarcastic, etc.)
        hf_client: Hugging Face client instance
        model: Text model to use

    Returns:
        Generated report text
    """
    from config import HF_TEXT_MODEL

    model = model or HF_TEXT_MODEL

    # Load and format prompt
    template = load_prompt(style)
    prompt = format_prompt(template, scene_data)

    # Get system prompt
    system_prompt = SYSTEM_PROMPTS.get(style)

    # Generate report
    logger.info(f"Generating {style} report for scene {scene_data.get('scene_id')}")
    try:
        report = hf_client.generate_text(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            max_tokens=150
        )
        logger.info(f"Generated {style} report: {len(report)} chars")
        return report
    except Exception as e:
        logger.warning(f"Text generation failed: {e}. Using local fallback.")
        return _local_fallback_report(scene_data, style)


def generate_all_reports(
    scene_data: dict,
    hf_client: HFClient,
    styles: list[str] = None,
    model: str = None
) -> dict[str, str]:
    """
    Generate reports in all available styles.

    Args:
        scene_data: Aggregated scene analysis
        hf_client: Hugging Face client instance
        styles: List of styles to generate (None = all)
        model: Text model to use

    Returns:
        Dictionary mapping style to report text
    """
    from config import REPORT_STYLES

    styles = styles or REPORT_STYLES
    reports = {}

    for style in styles:
        try:
            report = generate_report(scene_data, style, hf_client, model)
            reports[style] = report
        except Exception as e:
            logger.error(f"Error generating {style} report: {e}")
            reports[style] = f"Error generating report: {e}"

    return reports


def get_system_prompt(style: str) -> Optional[str]:
    """
    Get the system prompt for a given style.

    Args:
        style: Writing style

    Returns:
        System prompt or None
    """
    return SYSTEM_PROMPTS.get(style)


def _local_fallback_report(scene_data: dict, style: str) -> str:
    """Generate a simple local report when API is unavailable. Max 50 words, 2 lines."""
    summary = scene_data.get("summary", "video content detected")
    # Trim summary to ~30 words max
    words = summary.split()[:30]
    summary = " ".join(words)

    fallbacks = {
        "formal": f"{summary}.",
        "sarcastic": f"Oh look, {summary.lower()}. How absolutely riveting.",
        "humorous_tech": f"Scene loaded: {summary.lower()}. No bugs found, just vibes.",
        "humorous_non_tech": f"So basically: {summary.lower()}. Peak entertainment right there.",
        "jargon": f"Assessment: {summary.lower()}. Confidence metric: 70%.",
    }

    return fallbacks.get(style, summary)


def aggregate_activities(scene_analyses: dict) -> str:
    """
    Aggregate activities from all scenes into a single summary string.

    Args:
        scene_analyses: Dictionary mapping scene_id to scene analysis data

    Returns:
        Combined activities text from all scenes
    """
    activities_list = []
    for scene_id, scene_data in scene_analyses.items():
        activity = scene_data.get("activities", "")
        summary = scene_data.get("summary", "")
        if activity:
            activities_list.append(f"Scene {scene_id}: {activity}")
        elif summary:
            activities_list.append(f"Scene {scene_id}: {summary}")

    return "\n".join(activities_list) if activities_list else "No activities detected"


def generate_video_summary_reports(
    scene_analyses: dict,
    hf_client: HFClient,
    styles: list[str] = None,
    model: str = None
) -> dict[str, str]:
    """
    Generate 2-line reports for each style based on activities from ALL scenes.

    Args:
        scene_analyses: Dictionary mapping scene_id to scene analysis data
        hf_client: Hugging Face/Fireworks client instance
        styles: List of styles to generate (None = all)
        model: Text model to use

    Returns:
        Dictionary mapping style to 2-line report text
    """
    from config import REPORT_STYLES

    styles = styles or REPORT_STYLES

    # Aggregate activities from all scenes
    activities_summary = aggregate_activities(scene_analyses)

    # Create a combined scene_data dict for the prompt
    combined_data = {
        "activities": activities_summary,
        "summary": activities_summary,
        "scene_id": "all",
        "scene_type": "video",
        "location": "multiple scenes",
        "objects": [],
        "weather": "unknown",
        "time_of_day": "unknown",
        "environment": "unknown"
    }

    reports = {}
    for style in styles:
        try:
            report = generate_report(combined_data, style, hf_client, model)
            reports[style] = report
        except Exception as e:
            logger.error(f"Error generating {style} report: {e}")
            reports[style] = f"Error generating report: {e}"

    return reports
