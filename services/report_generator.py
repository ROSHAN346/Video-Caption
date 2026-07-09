"""
Report Generator Service

Generates multi-tone reports using Hugging Face text models.
"""

import logging
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.fireworks_client import FireworksClient
from services.prompt_loader import load_prompt, format_prompt

logger = logging.getLogger(__name__)

# System prompts for each style
SYSTEM_PROMPTS = {
    "formal": "You are a professional report writer. Output ONLY the final 2-line summary. No reasoning, no explanation, no chain-of-thought. Just the 2 sentences.",
    "sarcastic": "You are a witty, sarcastic commentator. Output ONLY the final 2-line sarcastic summary. No reasoning, no explanation, no chain-of-thought. Just the 2 sentences.",
    "humorous_tech": "You are a programmer who finds tech hilarious. Output ONLY the final 2-line funny summary with programming references. No reasoning, no explanation, no chain-of-thought. Just the 2 sentences.",
    "humorous_non_tech": "You are a stand-up comedian. Output ONLY the final 2-line funny summary. No reasoning, no explanation, no chain-of-thought. Just the 2 sentences.",
}


def generate_report(
    scene_data: dict,
    style: str,
    hf_client: FireworksClient,
    model: str = None
) -> str:
    """
    Generate a single report in the specified style.

    Args:
        scene_data: Aggregated scene analysis
        style: Writing style (formal, sarcastic, etc.)
        hf_client: Fireworks client instance
        model: Text model to use

    Returns:
        Generated report text
    """
    from config import FIREWORKS_TEXT_MODEL, GEMINI_TEXT_MODEL, AI_PROVIDER
    default_model = GEMINI_TEXT_MODEL if AI_PROVIDER == "gemini" else FIREWORKS_TEXT_MODEL
    model = model or default_model

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
            max_tokens=100
        )
        # Clean up any chain-of-thought reasoning
        report = _clean_report_output(report)
        logger.info(f"Generated {style} report: {len(report)} chars")
        return report
    except Exception as e:
        logger.warning(f"Text generation failed: {e}. Using local fallback.")
        return _local_fallback_report(scene_data, style)


def generate_all_reports(
    scene_data: dict,
    hf_client: FireworksClient,
    styles: list[str] = None,
    model: str = None
) -> dict[str, str]:
    """
    Generate reports in all available styles.

    Args:
        scene_data: Aggregated scene analysis
        hf_client: Fireworks client instance
        styles: List of styles to generate (None = all)
        model: Text model to use

    Returns:
        Dictionary mapping style to report text
    """
    from config import REPORT_STYLES

    styles = styles or REPORT_STYLES
    reports = {}

    def _gen_one(style):
        try:
            report = generate_report(scene_data, style, hf_client, model)
            return style, report
        except Exception as e:
            logger.error(f"Error generating {style} report: {e}")
            return style, f"Error generating report: {e}"

    with ThreadPoolExecutor(max_workers=min(len(styles), 4)) as executor:
        futures = {executor.submit(_gen_one, style): style for style in styles}
        for future in as_completed(futures):
            style, report = future.result()
            reports[style] = report

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


def _clean_report_output(report: str) -> str:
    """Strip chain-of-thought reasoning, return only the final 2-line summary."""
    if not report or not report.strip():
        return "No content generated."

    report = report.strip()

    # For reasoning models, the actual answer is at the very end
    # Split into sentences and take last 2
    import re
    sentences = re.split(r'(?<=[.!?])\s+', report)

    # Take last 2 sentences (the actual answer for reasoning models)
    if len(sentences) >= 2:
        result = " ".join(sentences[-2:])
    elif sentences:
        result = sentences[-1]
    else:
        result = report

    # Final cleanup: ensure reasonable length
    if len(result) > 300:
        # Take last 300 chars at sentence boundary
        result = result[-300:]
        # Find first sentence start
        for i, c in enumerate(result):
            if c.isupper() and i > 0:
                result = result[i:]
                break

    return result


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
    hf_client: FireworksClient,
    styles: list[str] = None,
    model: str = None
) -> dict[str, str]:
    """
    Generate 2-line reports for each style based on activities from ALL scenes.

    Args:
        scene_analyses: Dictionary mapping scene_id to scene analysis data
        hf_client: Fireworks client instance
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

    def _gen_one(style):
        try:
            report = generate_report(combined_data, style, hf_client, model)
            return style, report
        except Exception as e:
            logger.error(f"Error generating {style} report: {e}")
            return style, f"Error generating report: {e}"

    with ThreadPoolExecutor(max_workers=min(len(styles), 4)) as executor:
        futures = {executor.submit(_gen_one, style): style for style in styles}
        for future in as_completed(futures):
            style, report = future.result()
            reports[style] = report

    return reports
