"""
Scene Aggregator Service

Merges multiple per-frame analyses into a unified scene-level
analysis using consensus-based aggregation.
"""

import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


def aggregate_scene_analyses(analyses: list[dict]) -> dict:
    """
    Aggregate multiple frame analyses into a single scene analysis.

    Args:
        analyses: List of per-frame analysis dictionaries

    Returns:
        Aggregated scene analysis dictionary
    """
    if not analyses:
        logger.warning("No analyses to aggregate")
        return _empty_scene_analysis()

    if len(analyses) == 1:
        return analyses[0]

    scene_id = analyses[0].get("scene_id", 0)

    aggregated = {
        "scene_id": scene_id,
        "scene_type": consensus_field([a.get("scene_type", "unknown") for a in analyses]),
        "location": consensus_field([a.get("location", "unknown") for a in analyses]),
        "people": merge_people_descriptions([a.get("people", "unknown") for a in analyses]),
        "objects": merge_lists([a.get("objects", []) for a in analyses]),
        "vehicles": merge_lists([a.get("vehicles", []) for a in analyses]),
        "animals": merge_lists([a.get("animals", []) for a in analyses]),
        "activities": merge_activities([a.get("activities", "unknown") for a in analyses]),
        "weather": consensus_field([a.get("weather", "unknown") for a in analyses]),
        "time_of_day": consensus_field([a.get("time_of_day", "unknown") for a in analyses]),
        "environment": consensus_field([a.get("environment", "unknown") for a in analyses]),
        "risk_level": merge_risk_levels([a.get("risk_level", "unknown") for a in analyses]),
        "confidence": calculate_overall_confidence(analyses),
        "summary": merge_summaries([a.get("summary", "") for a in analyses]),
        "frame_count": len(analyses)
    }

    logger.info(f"Aggregated {len(analyses)} frames for scene {scene_id}")
    return aggregated


def consensus_field(values: list) -> str:
    """
    Find the most common value (majority vote).

    Args:
        values: List of values (strings or lists)

    Returns:
        Most common value as string
    """
    # Filter out unknowns and convert lists to strings
    valid_values = []
    for v in values:
        if v and v != "unknown":
            if isinstance(v, list):
                valid_values.append(", ".join(str(item) for item in v))
            else:
                valid_values.append(str(v))

    if not valid_values:
        return "unknown"

    counter = Counter(valid_values)
    return counter.most_common(1)[0][0]


def merge_lists(lists: list[list]) -> list:
    """
    Merge multiple lists, removing duplicates.

    Args:
        lists: List of lists to merge

    Returns:
        Deduplicated merged list
    """
    merged = []
    seen = set()

    for lst in lists:
        if isinstance(lst, list):
            for item in lst:
                item_str = str(item).lower().strip()
                if item_str not in seen and item_str:
                    seen.add(item_str)
                    merged.append(str(item))

    return merged


def merge_people_descriptions(descriptions: list) -> str:
    """
    Merge multiple people descriptions.

    Args:
        descriptions: List of people descriptions (strings or lists)

    Returns:
        Combined description
    """
    # Filter out unknowns and empties, convert lists to strings
    valid = []
    for d in descriptions:
        if d and d != "unknown" and d != "none":
            if isinstance(d, list):
                valid.append(", ".join(str(item) for item in d))
            else:
                valid.append(str(d))

    if not valid:
        return "none visible"

    # Take the most detailed description
    return max(valid, key=len)


def merge_activities(activities: list[str]) -> str:
    """
    Merge multiple activity descriptions.

    Args:
        activities: List of activity descriptions

    Returns:
        Combined activities
    """
    valid = [a for a in activities if a and a != "unknown"]

    if not valid:
        return "unknown"

    # Combine unique activities
    seen = set()
    merged = []

    for activity in valid:
        if isinstance(activity, list):
            for item in activity:
                item = str(item).strip().lower()
                if item and item not in seen:
                    seen.add(item)
                    merged.append(item)
        else:
            parts = str(activity).replace(";", ",").split(",")
            for part in parts:
                part = part.strip().lower()
                if part and part not in seen:
                    seen.add(part)
                    merged.append(part)

    return ", ".join(merged) if merged else valid[0]


def merge_risk_levels(risks: list) -> str:
    """
    Merge risk levels, taking the highest severity.

    Args:
        risks: List of risk level descriptions (strings or lists)

    Returns:
        Highest risk level with reason
    """
    risk_priority = {"high": 3, "medium": 2, "low": 1, "unknown": 0}

    highest_risk = "unknown"
    highest_priority = 0
    highest_reason = ""

    for risk in risks:
        if not risk or risk == "unknown":
            continue

        if isinstance(risk, list):
            risk = ", ".join(str(item) for item in risk)

        risk_lower = str(risk).lower()

        for level in ["high", "medium", "low"]:
            if level in risk_lower:
                priority = risk_priority[level]
                if priority > highest_priority:
                    highest_priority = priority
                    highest_risk = level
                    if "-" in risk:
                        highest_reason = risk.split("-", 1)[1].strip()
                    elif ":" in risk:
                        highest_reason = risk.split(":", 1)[1].strip()
                    else:
                        highest_reason = risk
                break

    if highest_risk == "unknown":
        return "unknown"

    return f"{highest_risk} - {highest_reason}" if highest_reason else highest_risk


def calculate_overall_confidence(analyses: list[dict]) -> float:
    """
    Calculate average confidence across all analyses.

    Args:
        analyses: List of analysis dictionaries

    Returns:
        Average confidence (0.0-1.0)
    """
    confidences = [
        a.get("confidence", 0.0)
        for a in analyses
        if isinstance(a.get("confidence"), (int, float))
    ]

    if not confidences:
        return 0.0

    return round(sum(confidences) / len(confidences), 2)


def merge_summaries(summaries: list) -> str:
    """
    Get the best summary from the list (longest unique one).

    Args:
        summaries: List of summary strings (or lists)

    Returns:
        Best single summary
    """
    valid = []
    for s in summaries:
        if s:
            if isinstance(s, list):
                s = " ".join(str(item) for item in s)
            s = str(s).strip()
            if s and s != "unknown":
                valid.append(s)

    if not valid:
        return "No summary available"

    # Remove duplicates and pick the longest (most descriptive)
    seen = set()
    unique = []
    for s in valid:
        s_clean = s.strip().lower()
        if s_clean not in seen:
            seen.add(s_clean)
            unique.append(s.strip())

    if not unique:
        return "No summary available"

    # Return the longest summary (most descriptive)
    return max(unique, key=len)


def _empty_scene_analysis() -> dict:
    """Return empty scene analysis structure."""
    return {
        "scene_id": 0,
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
        "summary": "No analysis available",
        "frame_count": 0
    }


def aggregate_by_scene(
    analyses: list[dict],
    scenes: list
) -> dict[int, dict]:
    """
    Group analyses by scene and aggregate each.

    Args:
        analyses: List of per-frame analyses (each with scene_id)
        scenes: List of Scene objects

    Returns:
        Dictionary mapping scene_id to aggregated analysis
    """
    # Group by scene_id
    scene_groups = {}
    for analysis in analyses:
        scene_id = analysis.get("scene_id", 0)
        if scene_id not in scene_groups:
            scene_groups[scene_id] = []
        scene_groups[scene_id].append(analysis)

    # Aggregate each scene
    aggregated = {}
    for scene_id, group_analyses in scene_groups.items():
        aggregated[scene_id] = aggregate_scene_analyses(group_analyses)

    return aggregated
