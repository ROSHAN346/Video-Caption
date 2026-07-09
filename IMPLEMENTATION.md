# Implementation Guide — AI Image Analysis & Multi-Tone Report Generation

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Phase 1: Foundation Setup](#2-phase-1-foundation-setup)
3. [Phase 2: HF Client Service](#3-phase-2-hf-client-service)
4. [Phase 3: Image Analysis Service](#4-phase-3-image-analysis-service)
5. [Phase 4: Scene Aggregation](#5-phase-4-scene-aggregation)
6. [Phase 5: Prompt Templates](#6-phase-5-prompt-templates)
7. [Phase 6: Report Generation](#7-phase-6-report-generation)
8. [Phase 7: Report Caching](#8-phase-7-report-caching)
9. [Phase 8: PDF Generation](#9-phase-8-pdf-generation)
10. [Phase 9: CLI Integration](#10-phase-9-cli-integration)
11. [Phase 10: Streamlit UI Integration](#11-phase-10-streamlit-ui-integration)
12. [Phase 11: Testing](#12-phase-11-testing)
13. [Phase 12: Documentation](#13-phase-12-documentation)
14. [Appendix: Complete File Listing](#14-appendix-complete-file-listing)

---

## 1. Prerequisites

### 1.1 Hugging Face Account Setup

1. Go to https://huggingface.co
2. Create an account or sign in
3. Navigate to Settings > Access Tokens
4. Click "New token"
5. Select "read" permissions
6. Copy the token (starts with `hf_`)

### 1.2 Environment Setup

```bash
# Navigate to project directory
cd video-amd-main

# Create .env file for API token
echo "HF_API_TOKEN=hf_your_token_here" > .env

# Verify .env is in .gitignore
cat .gitignore | grep .env
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Phase 1: Foundation Setup

### 2.1 Update `config.py`

**File:** `config.py`

**Add the following configuration keys:**

```python
import os
from dotenv import load_dotenv

load_dotenv()

# ... existing config ...

# --- Hugging Face Configuration ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
HF_TEXT_MODEL = "Qwen/Qwen3-8B-Instruct"
HF_VISION_PROVIDER = "nebius"
HF_TEXT_PROVIDER = "nebius"

# --- Report Generation ---
REPORT_STYLES = [
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
    "jargon"
]
DEFAULT_REPORT_STYLE = "formal"
REPORT_CACHE_ENABLED = True
REPORT_LANGUAGE = "en"

# --- Scene JSON Schema ---
SCENE_JSON_FIELDS = [
    "scene_id", "scene_type", "location", "people", "objects",
    "vehicles", "animals", "activities", "weather", "time_of_day",
    "environment", "risk_level", "confidence", "summary"
]
```

### 2.2 Update `requirements.txt`

**File:** `requirements.txt`

**Add new dependencies:**

```txt
opencv-python>=4.5.0
numpy>=1.20.0
torch>=1.9.0
Pillow>=8.0.0
scenedetect>=0.5.0
git+https://github.com/openai/CLIP.git
streamlit>=1.28.0
huggingface_hub>=0.25.0
fpdf2>=2.8.0
python-dotenv>=1.0.0
```

### 2.3 Create Directory Structure

```bash
# Create new directories
mkdir -p services
mkdir -p prompts
mkdir -p analysis
mkdir -p reports

# Create __init__.py for services package
touch services/__init__.py
```

### 2.4 Create `.env` Template

**File:** `.env.example`

```
# Hugging Face API Token
# Get yours at: https://huggingface.co/settings/tokens
HF_API_TOKEN=hf_your_token_here

# Optional: Override default models
# HF_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
# HF_TEXT_MODEL=Qwen/Qwen3-8B-Instruct
```

---

## 3. Phase 2: HF Client Service

### 3.1 Create `services/hf_client.py`

**File:** `services/hf_client.py`

**Purpose:** Centralized Hugging Face API communication

```python
"""
Hugging Face API Client Service

Provides a unified interface for communicating with Hugging Face
Inference API for both vision and text models.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class HFClient:
    """Wrapper around Hugging Face InferenceClient."""

    def __init__(self, api_token: str, provider: str = "nebius"):
        """
        Initialize the HF client.

        Args:
            api_token: Hugging Face API token
            provider: Inference provider (nebius, together, fireworks)
        """
        if not api_token:
            raise ValueError(
                "HF_API_TOKEN is required. "
                "Get yours at: https://huggingface.co/settings/tokens"
            )

        self.api_token = api_token
        self.provider = provider
        self.client = InferenceClient(token=api_token, provider=provider)
        logger.info(f"HFClient initialized with provider: {provider}")

    def analyze_image(
        self,
        image_path: str,
        prompt: str,
        model: str,
        max_tokens: int = 1024
    ) -> str:
        """
        Send image + prompt to a vision model.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for analysis
            model: Model identifier (e.g., Qwen/Qwen2.5-VL-7B-Instruct)
            max_tokens: Maximum tokens in response

        Returns:
            Model response text
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read and encode image
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Determine image MIME type
        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
        mime_type = mime_map.get(suffix, "jpeg")
        image_url = f"data:image/{mime_type};base64,{image_b64}"

        # Build message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        # Call API
        logger.info(f"Analyzing image: {image_path.name} with model: {model}")
        response = self.client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )

        result = response.choices[0].message.content
        logger.debug(f"Image analysis response length: {len(result)} chars")
        return result

    def analyze_image_base64(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        mime_type: str = "jpeg",
        max_tokens: int = 1024
    ) -> str:
        """
        Analyze image from raw bytes.

        Args:
            image_bytes: Raw image bytes
            prompt: Text prompt for analysis
            model: Model identifier
            mime_type: Image MIME type
            max_tokens: Maximum tokens in response

        Returns:
            Model response text
        """
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/{mime_type};base64,{image_b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        response = self.client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def generate_text(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate text using a language model.

        Args:
            prompt: User prompt
            model: Model identifier
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        logger.info(f"Generating text with model: {model}")
        response = self.client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )

        result = response.choices[0].message.content
        logger.debug(f"Text generation response length: {len(result)} chars")
        return result

    def validate_connection(self) -> bool:
        """
        Validate the API connection works.

        Returns:
            True if connection is valid
        """
        try:
            # Simple test with a small model
            response = self.client.chat_completion(
                model="gpt2",
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


# Singleton instance for convenience
_client_instance: Optional[HFClient] = None


def get_hf_client(api_token: str = None, provider: str = None) -> HFClient:
    """
    Get or create HF client singleton.

    Args:
        api_token: API token (uses config default if None)
        provider: Provider name (uses config default if None)

    Returns:
        HFClient instance
    """
    global _client_instance

    if _client_instance is None:
        from config import HF_API_TOKEN, HF_VISION_PROVIDER

        token = api_token or HF_API_TOKEN
        prov = provider or HF_VISION_PROVIDER

        _client_instance = HFClient(token, prov)

    return _client_instance
```

---

## 4. Phase 3: Image Analysis Service

### 4.1 Create `services/image_analyzer.py`

**File:** `services/image_analyzer.py`

**Purpose:** Convert keyframes to structured JSON descriptions

```python
"""
Image Analysis Service

Analyzes keyframes using Hugging Face Vision models and produces
structured JSON descriptions of each frame.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from services.hf_client import HFClient

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


def analyze_keyframe(
    image_path: str,
    hf_client: HFClient,
    model: str = None,
    scene_id: int = 0,
    frame_index: int = 0
) -> dict:
    """
    Analyze a single keyframe and return structured JSON.

    Args:
        image_path: Path to the keyframe image
        hf_client: Hugging Face client instance
        model: Vision model to use (uses config default if None)
        scene_id: Scene ID for metadata
        frame_index: Frame index for metadata

    Returns:
        Dictionary with structured analysis
    """
    from config import HF_VISION_MODEL

    model = model or HF_VISION_MODEL
    image_path = Path(image_path)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return _empty_analysis(scene_id, frame_index)

    try:
        # Call vision model
        response = hf_client.analyze_image(
            image_path=str(image_path),
            prompt=ANALYSIS_PROMPT,
            model=model,
            max_tokens=1024
        )

        # Parse JSON response
        analysis = _parse_json_response(response)

        # Add metadata
        analysis["scene_id"] = scene_id
        analysis["frame_index"] = frame_index
        analysis["image_path"] = image_path.name

        logger.info(f"Analyzed frame {frame_index} in scene {scene_id}")
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing frame {frame_index}: {e}")
        return _empty_analysis(scene_id, frame_index)


def analyze_keyframes_batch(
    image_paths: list[str],
    hf_client: HFClient,
    model: str = None,
    scene_id: int = 0
) -> list[dict]:
    """
    Analyze a batch of keyframes.

    Args:
        image_paths: List of image paths
        hf_client: Hugging Face client instance
        model: Vision model to use
        scene_id: Scene ID for metadata

    Returns:
        List of analysis dictionaries
    """
    analyses = []

    for idx, image_path in enumerate(image_paths):
        analysis = analyze_keyframe(
            image_path=image_path,
            hf_client=hf_client,
            model=model,
            scene_id=scene_id,
            frame_index=idx
        )
        analyses.append(analysis)

    return analyses


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
```

---

## 5. Phase 4: Scene Aggregation

### 5.1 Create `services/scene_aggregator.py`

**File:** `services/scene_aggregator.py`

**Purpose:** Merge multiple keyframe analyses into a single scene representation

```python
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


def consensus_field(values: list[str]) -> str:
    """
    Find the most common value (majority vote).

    Args:
        values: List of string values

    Returns:
        Most common value
    """
    # Filter out unknowns
    valid_values = [v for v in values if v and v != "unknown"]

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


def merge_people_descriptions(descriptions: list[str]) -> str:
    """
    Merge multiple people descriptions.

    Args:
        descriptions: List of people descriptions

    Returns:
        Combined description
    """
    # Filter out unknowns and empties
    valid = [d for d in descriptions if d and d != "unknown" and d != "none"]

    if not valid:
        return "none visible"

    # Take the most detailed description
    # (longest is usually most detailed)
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
        # Split by common delimiters
        parts = activity.replace(";", ",").split(",")
        for part in parts:
            part = part.strip().lower()
            if part and part not in seen:
                seen.add(part)
                merged.append(part)

    return ", ".join(merged) if merged else valid[0]


def merge_risk_levels(risks: list[str]) -> str:
    """
    Merge risk levels, taking the highest severity.

    Args:
        risks: List of risk level descriptions

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

        risk_lower = risk.lower()

        # Extract risk level and reason
        for level in ["high", "medium", "low"]:
            if level in risk_lower:
                priority = risk_priority[level]
                if priority > highest_priority:
                    highest_priority = priority
                    highest_risk = level
                    # Extract reason (text after dash or colon)
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


def merge_summaries(summaries: list[str]) -> str:
    """
    Merge multiple summaries into a single paragraph.

    Args:
        summaries: List of summary strings

    Returns:
        Combined summary
    """
    valid = [s for s in summaries if s and s.strip()]

    if not valid:
        return "No summary available"

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for s in valid:
        s_clean = s.strip().lower()
        if s_clean not in seen:
            seen.add(s_clean)
            unique.append(s.strip())

    if len(unique) == 1:
        return unique[0]

    # Combine into paragraph
    return " ".join(unique)


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
```

---

## 6. Phase 5: Prompt Templates

### 6.1 Create Prompt Files

**Directory:** `prompts/`

#### `prompts/formal.txt`

```
You are a professional report writer creating formal documentation.
Your writing style should be:
- Professional and factual
- Clear and concise
- Objective and unbiased
- Using proper grammar and punctuation
- Avoiding colloquialisms and slang

Analyze the following scene data and write a formal report.

Scene Data:
{scene_data}

Write a formal, professional report based on this scene analysis.
The report should be suitable for official documentation.
```

#### `prompts/sarcastic.txt`

```
You are a witty, sarcastic commentator with a dry sense of humor.
Your writing style should be:
- Full of dry irony and sarcasm
- Clever but still conveying facts accurately
- Like a tired detective narrating their cases
- Subtly mocking while remaining informative
- Using understatement and overstatement for comedic effect

Analyze the following scene data and write a sarcastic report.

Scene Data:
{scene_data}

Write a sarcastic, witty report based on this scene analysis.
Remember: the facts must still be accurate!
```

#### `prompts/humorous_tech.txt`

```
You are a programmer who finds everything related to tech hilarious.
Your writing style should be:
- Full of programming jokes and developer humor
- Using tech metaphors and analogies
- Referencing Stack Overflow, GitHub, and coding culture
- Making fun of common developer experiences
- Including puns about code, bugs, and algorithms

Analyze the following scene data and write a humorous tech-themed report.

Scene Data:
{scene_data}

Write a report that a software developer would find funny.
Include at least 3 programming references or jokes.
```

#### `prompts/humorous_non_tech.txt`

```
You are a stand-up comedian who finds everyday situations hilarious.
Your writing style should be:
- Relatable everyday humor
- Observational comedy
- Finding the absurd in the mundane
- Using exaggeration and comedic timing
- Making people laugh while informing them

Analyze the following scene data and write a funny report.

Scene Data:
{scene_data}

Write a report that anyone would find entertaining and funny.
Make the mundane sound exciting and the exciting sound mundane.
```

#### `prompts/jargon.txt`

```
You are a domain expert who speaks exclusively in professional jargon.
Your writing style should be:
- Heavy use of domain-specific terminology
- Mixing medical, legal, cybersecurity, engineering, and finance jargon
- Using acronyms and technical terms
- Sounding extremely knowledgeable but slightly inaccessible
- Like reading a technical manual written by multiple departments

Analyze the following scene data and write a jargon-heavy report.

Scene Data:
{scene_data}

Write a report overloaded with professional jargon from multiple fields.
The reader should need a glossary to understand it.
```

### 6.2 Create `services/prompt_loader.py`

**File:** `services/prompt_loader.py`

```python
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
```

---

## 7. Phase 6: Report Generation

### 7.1 Create `services/report_generator.py`

**File:** `services/report_generator.py`

```python
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
    "formal": "You are a professional report writer. Write formally and factually.",
    "sarcastic": "You are a witty, sarcastic commentator. Be clever but accurate.",
    "humorous_tech": "You are a programmer who finds tech hilarious. Use programming jokes.",
    "humorous_non_tech": "You are a stand-up comedian. Find humor in everyday situations.",
    "jargon": "You are a domain expert who speaks in heavy professional jargon."
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
    report = hf_client.generate_text(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        max_tokens=2048
    )

    logger.info(f"Generated {style} report: {len(report)} chars")
    return report


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
```

---

## 8. Phase 7: Report Caching

### 8.1 Create `services/report_cache.py`

**File:** `services/report_cache.py`

```python
"""
Report Cache Service

Provides file-based caching for generated reports to avoid
regeneration for already-processed scenes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ReportCache:
    """File-based report cache."""

    def __init__(self, base_dir: str = "reports"):
        """
        Initialize cache.

        Args:
            base_dir: Base directory for cached reports
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, scene_id: int, style: str) -> Path:
        """
        Get the file path for a cached report.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            Path to cached report file
        """
        scene_dir = self.base_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir / f"{style}.md"

    def get_pdf_path(self, scene_id: int, style: str) -> Path:
        """
        Get the file path for a cached PDF report.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            Path to cached PDF file
        """
        scene_dir = self.base_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir / f"{style}.pdf"

    def get_cache_metadata_path(self, scene_id: int) -> Path:
        """
        Get the file path for cache metadata.

        Args:
            scene_id: Scene ID

        Returns:
            Path to metadata file
        """
        scene_dir = self.base_dir / f"scene_{scene_id:03d}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir / "cache.json"

    def is_cached(self, scene_id: int, style: str) -> bool:
        """
        Check if a report is cached.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            True if report exists in cache
        """
        cache_path = self.get_cache_path(scene_id, style)
        return cache_path.exists()

    def get_cached_report(self, scene_id: int, style: str) -> Optional[str]:
        """
        Retrieve a cached report.

        Args:
            scene_id: Scene ID
            style: Report style

        Returns:
            Cached report text or None
        """
        cache_path = self.get_cache_path(scene_id, style)

        if not cache_path.exists():
            return None

        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    def cache_report(
        self,
        scene_id: int,
        style: str,
        report: str,
        metadata: dict = None
    ):
        """
        Cache a generated report.

        Args:
            scene_id: Scene ID
            style: Report style
            report: Report text
            metadata: Optional metadata to store
        """
        # Save report
        cache_path = self.get_cache_path(scene_id, style)
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Update metadata
        self._update_metadata(scene_id, style, metadata)

        logger.info(f"Cached {style} report for scene {scene_id}")

    def _update_metadata(self, scene_id: int, style: str, metadata: dict = None):
        """Update cache metadata."""
        meta_path = self.get_cache_metadata_path(scene_id)

        # Load existing metadata
        existing = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                existing = json.load(f)

        # Update
        existing["scene_id"] = scene_id
        existing["generated_at"] = datetime.now().isoformat()

        if "styles" not in existing:
            existing["styles"] = []

        if style not in existing["styles"]:
            existing["styles"].append(style)

        if metadata:
            existing.update(metadata)

        # Save
        with open(meta_path, "w") as f:
            json.dump(existing, f, indent=2)

    def clear_cache(self, scene_id: int = None):
        """
        Clear cache for a specific scene or all scenes.

        Args:
            scene_id: Scene ID to clear (None = clear all)
        """
        if scene_id is not None:
            scene_dir = self.base_dir / f"scene_{scene_id:03d}"
            if scene_dir.exists():
                import shutil
                shutil.rmtree(scene_dir)
                logger.info(f"Cleared cache for scene {scene_id}")
        else:
            import shutil
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all report cache")

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_reports = 0
        total_scenes = 0
        total_size = 0

        if self.base_dir.exists():
            for scene_dir in self.base_dir.iterdir():
                if scene_dir.is_dir() and scene_dir.name.startswith("scene_"):
                    total_scenes += 1
                    for report_file in scene_dir.glob("*.md"):
                        total_reports += 1
                        total_size += report_file.stat().st_size

        return {
            "total_scenes": total_scenes,
            "total_reports": total_reports,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
```

---

## 9. Phase 8: PDF Generation

### 9.1 Create `services/pdf_generator.py`

**File:** `services/pdf_generator.py`

```python
"""
PDF Generator Service

Converts Markdown reports to PDF format with embedded keyframe images.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)


class ReportPDF(FPDF):
    """Custom PDF class for report generation."""

    def header(self):
        """Add header to each page."""
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 10, "Video Keyframe Analysis Report", align="C")
        self.ln(10)

    def footer(self):
        """Add footer to each page."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def generate_report_pdf(
    scene_id: int,
    style: str,
    report: str,
    keyframe_path: str = None,
    scene_data: dict = None,
    output_path: str = None
) -> Path:
    """
    Generate a PDF report.

    Args:
        scene_id: Scene ID
        style: Report style
        report: Report text
        keyframe_path: Optional path to keyframe image
        scene_data: Optional scene analysis data
        output_path: Optional output path

    Returns:
        Path to generated PDF
    """
    if output_path is None:
        output_path = f"reports/scene_{scene_id:03d}/{style}.pdf"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create PDF
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Add keyframe image if provided
    if keyframe_path and Path(keyframe_path).exists():
        _add_keyframe_page(pdf, keyframe_path, scene_data)

    # Add report content
    _add_report_content(pdf, scene_id, style, report, scene_data)

    # Save
    pdf.output(str(output_path))
    logger.info(f"Generated PDF: {output_path}")

    return output_path


def _add_keyframe_page(pdf: ReportPDF, keyframe_path: str, scene_data: dict = None):
    """Add keyframe image page."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Scene Keyframe", ln=True)
    pdf.ln(5)

    # Add image
    try:
        pdf.image(keyframe_path, x=10, w=190)
        pdf.ln(10)
    except Exception as e:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"[Image could not be embedded: {e}]", ln=True)

    # Add scene metadata if available
    if scene_data:
        _add_metadata_table(pdf, scene_data)

    pdf.add_page()


def _add_metadata_table(pdf: ReportPDF, scene_data: dict):
    """Add scene metadata table."""
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Scene Metadata", ln=True)

    pdf.set_font("Helvetica", "", 10)

    fields = [
        ("Scene ID", scene_data.get("scene_id", "N/A")),
        ("Scene Type", scene_data.get("scene_type", "N/A")),
        ("Location", scene_data.get("location", "N/A")),
        ("Risk Level", scene_data.get("risk_level", "N/A")),
        ("Confidence", f"{scene_data.get('confidence', 0):.2f}"),
    ]

    for label, value in fields:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(50, 8, f"{label}:", border=1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, str(value), border=1, ln=True)

    pdf.ln(5)


def _add_report_content(
    pdf: ReportPDF,
    scene_id: int,
    style: str,
    report: str,
    scene_data: dict = None
):
    """Add report content."""
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"Scene {scene_id} - {style.replace('_', ' ').title()} Report", ln=True)
    pdf.ln(5)

    # Metadata
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Style: {style}", ln=True)
    pdf.ln(5)

    # Report content
    pdf.set_font("Helvetica", "", 11)

    # Split report into paragraphs and add them
    paragraphs = report.split("\n\n")

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # Check if it's a heading (starts with #)
            if paragraph.startswith("#"):
                pdf.set_font("Helvetica", "B", 13)
                paragraph = paragraph.lstrip("#").strip()
                pdf.multi_cell(0, 8, paragraph)
                pdf.set_font("Helvetica", "", 11)
            else:
                pdf.multi_cell(0, 6, paragraph)
            pdf.ln(3)


def markdown_to_pdf(markdown_text: str, output_path: str) -> Path:
    """
    Convert markdown text to PDF.

    Args:
        markdown_text: Markdown formatted text
        output_path: Output PDF path

    Returns:
        Path to generated PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Parse and add markdown content
    lines = markdown_text.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            pdf.ln(3)
            continue

        # Handle headings
        if line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, line[4:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 8, line[3:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 8, line[2:])
            pdf.set_font("Helvetica", "", 11)
        elif line.startswith("- "):
            # Bullet point
            pdf.cell(10, 6, "•")
            pdf.multi_cell(0, 6, line[2:])
        else:
            pdf.multi_cell(0, 6, line)

    pdf.output(str(output_path))
    return output_path
```

---

## 10. Phase 9: CLI Integration

### 10.1 Update `main.py`

**File:** `main.py`

**Add the following imports and logic:**

```python
# Add at the top of main.py
import argparse

# Add after existing imports
from config import (
    HF_API_TOKEN, REPORT_STYLES, DEFAULT_REPORT_STYLE,
    REPORT_CACHE_ENABLED
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Video Keyframe Extraction with AI Report Generation"
    )
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Generate AI analysis reports"
    )
    parser.add_argument(
        "--style",
        choices=REPORT_STYLES,
        default=DEFAULT_REPORT_STYLE,
        help=f"Report style (default: {DEFAULT_REPORT_STYLE})"
    )
    parser.add_argument(
        "--all-styles",
        action="store_true",
        help="Generate reports in all styles"
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation"
    )
    return parser.parse_args()


# Add report generation function
def generate_reports_pipeline(
    output_dir: Path,
    selected: list,
    scenes: list,
    video_name: str
):
    """Run the report generation pipeline."""
    from services.hf_client import get_hf_client
    from services.image_analyzer import analyze_keyframes_batch, save_analysis
    from services.scene_aggregator import aggregate_by_scene
    from services.report_generator import generate_all_reports
    from services.report_cache import ReportCache
    from services.pdf_generator import generate_report_pdf

    logger.info("=== START Report Generation ===")

    # Initialize services
    hf_client = get_hf_client()
    cache = ReportCache(str(output_dir / "reports"))

    # Analyze keyframes
    logger.info("Analyzing keyframes with vision model...")
    keyframe_paths = [
        str(output_dir / f"keyframe_{i:03d}.jpg")
        for i in range(len(selected))
    ]

    analyses = []
    for i, (path, sf) in enumerate(zip(keyframe_paths, selected)):
        analysis = analyze_keyframe(
            image_path=path,
            hf_client=hf_client,
            scene_id=sf.scene_id,
            frame_index=i
        )
        analyses.append(analysis)

        # Save individual analysis
        analysis_path = output_dir / "analysis" / f"scene_{sf.scene_id:03d}_frame_{i:03d}.json"
        save_analysis(analysis, str(analysis_path))

    # Aggregate by scene
    logger.info("Aggregating analyses by scene...")
    scene_analyses = aggregate_by_scene(analyses, scenes)

    # Generate reports
    logger.info("Generating multi-tone reports...")
    for scene_id, scene_data in scene_analyses.items():
        reports = generate_all_reports(scene_data, hf_client)

        for style, report in reports.items():
            # Cache report
            cache.cache_report(scene_id, style, report, {
                "vision_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "text_model": "Qwen/Qwen3-8B-Instruct"
            })

            # Generate PDF if enabled
            if not args.no_pdf:
                keyframe_path = str(output_dir / f"keyframe_000.jpg")  # First keyframe for scene
                pdf_path = cache.get_pdf_path(scene_id, style)
                generate_report_pdf(
                    scene_id=scene_id,
                    style=style,
                    report=report,
                    keyframe_path=keyframe_path,
                    scene_data=scene_data,
                    output_path=str(pdf_path)
                )

    logger.info("✅ Report generation complete")


# Update main() function to include report generation
def main():
    args = parse_args()

    # ... existing code ...

    # After keyframe extraction, add:
    if args.reports:
        generate_reports_pipeline(output_dir, selected, scenes, video_name)
```

---

## 11. Phase 10: Streamlit UI Integration

### 11.1 Update `app.py`

**File:** `app.py`

**Add new sidebar section and main content:**

```python
# Add at top of app.py
from config import (
    HF_API_TOKEN, HF_VISION_MODEL, HF_TEXT_MODEL,
    REPORT_STYLES, REPORT_CACHE_ENABLED
)

# Add new sidebar section
st.sidebar.subheader("🤖 AI Analysis")
enable_reports = st.sidebar.checkbox("Enable AI Reports", value=False)

if enable_reports:
    hf_token = st.sidebar.text_input(
        "Hugging Face API Token",
        value=HF_API_TOKEN,
        type="password",
        help="Get yours at: https://huggingface.co/settings/tokens"
    )

    vision_model = st.sidebar.selectbox(
        "Vision Model",
        ["Qwen/Qwen2.5-VL-7B-Instruct", "SmolVLM", "Florence-2"],
        index=0
    )

    text_model = st.sidebar.selectbox(
        "Text Model",
        ["Qwen/Qwen3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"],
        index=0
    )

    report_styles = st.sidebar.multiselect(
        "Report Styles",
        REPORT_STYLES,
        default=["formal", "sarcastic"]
    )

    generate_pdf = st.sidebar.checkbox("Generate PDFs", value=True)

# Add report generation button
if uploaded_file is not None and enable_reports:
    if st.button("📝 Generate AI Reports", type="secondary"):
        # Report generation logic here
        pass

# Add report display section
if "reports" in st.session_state:
    st.divider()
    st.header("📝 Generated Reports")

    # Tabs for each style
    tabs = st.tabs([s.replace("_", " ").title() for s in st.session_state.reports.keys()])

    for tab, (style, report) in zip(tabs, st.session_state.reports.items()):
        with tab:
            st.markdown(report)

            # Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    f"📄 Download {style}.md",
                    report,
                    file_name=f"{style}.md",
                    mime="text/markdown"
                )
            with col2:
                if generate_pdf:
                    pdf_path = f"reports/scene_001/{style}.pdf"
                    if Path(pdf_path).exists():
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                f"📥 Download {style}.pdf",
                                f.read(),
                                file_name=f"{style}.pdf",
                                mime="application/pdf"
                            )
```

---

## 12. Phase 11: Testing

### 12.1 Create Test Files

**Directory:** `tests/`

#### `tests/__init__.py`

```python
# Test package
```

#### `tests/test_hf_client.py`

```python
"""Tests for HF Client service."""
import pytest
from unittest.mock import Mock, patch
from services.hf_client import HFClient


def test_hf_client_init():
    """Test HF client initialization."""
    with patch('services.hf_client.InferenceClient'):
        client = HFClient("test_token", "nebius")
        assert client.api_token == "test_token"
        assert client.provider == "nebius"


def test_hf_client_init_no_token():
    """Test HF client fails without token."""
    with pytest.raises(ValueError):
        HFClient("")
```

#### `tests/test_image_analyzer.py`

```python
"""Tests for Image Analyzer service."""
import pytest
from unittest.mock import Mock, patch
from services.image_analyzer import _parse_json_response, _empty_analysis


def test_parse_json_response():
    """Test JSON parsing from model response."""
    response = '{"scene_type": "outdoor", "location": "street"}'
    result = _parse_json_response(response)
    assert result["scene_type"] == "outdoor"
    assert result["location"] == "street"


def test_parse_json_response_with_markdown():
    """Test JSON parsing with markdown code blocks."""
    response = '```json\n{"scene_type": "indoor"}\n```'
    result = _parse_json_response(response)
    assert result["scene_type"] == "indoor"


def test_empty_analysis():
    """Test empty analysis structure."""
    result = _empty_analysis(scene_id=1, frame_index=0)
    assert result["scene_id"] == 1
    assert result["frame_index"] == 0
    assert result["confidence"] == 0.0
```

#### `tests/test_scene_aggregator.py`

```python
"""Tests for Scene Aggregator service."""
import pytest
from services.scene_aggregator import (
    consensus_field, merge_lists, merge_risk_levels,
    calculate_overall_confidence
)


def test_consensus_field():
    """Test majority vote consensus."""
    values = ["outdoor", "outdoor", "indoor", "outdoor"]
    assert consensus_field(values) == "outdoor"


def test_consensus_field_with_unknowns():
    """Test consensus filters unknowns."""
    values = ["unknown", "outdoor", "unknown"]
    assert consensus_field(values) == "outdoor"


def test_merge_lists():
    """Test list merging with deduplication."""
    lists = [["car", "bike"], ["car", "truck"], ["bike"]]
    result = merge_lists(lists)
    assert "car" in result
    assert "bike" in result
    assert "truck" in result
    assert len(result) == 3


def test_merge_risk_levels():
    """Test risk level merging."""
    risks = ["low - safe", "medium - crowded", "low - normal"]
    result = merge_risk_levels(risks)
    assert "medium" in result


def test_calculate_overall_confidence():
    """Test confidence calculation."""
    analyses = [
        {"confidence": 0.8},
        {"confidence": 0.9},
        {"confidence": 0.7}
    ]
    result = calculate_overall_confidence(analyses)
    assert result == 0.8
```

#### `tests/test_prompt_loader.py`

```python
"""Tests for Prompt Loader service."""
import pytest
from pathlib import Path
from services.prompt_loader import load_prompt, format_prompt, get_available_styles


def test_load_prompt():
    """Test loading a prompt template."""
    # Create test prompt
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True)
    (prompts_dir / "test_style.txt").write_text("Test prompt: {scene_data}")

    template = load_prompt("test_style")
    assert "{scene_data}" in template

    # Cleanup
    (prompts_dir / "test_style.txt").unlink()


def test_format_prompt():
    """Test prompt formatting."""
    template = "Scene type: {scene_data}"
    scene_data = {"scene_type": "outdoor"}

    result = format_prompt(template, scene_data)
    assert "outdoor" in result


def test_get_available_styles():
    """Test getting available styles."""
    styles = get_available_styles()
    assert isinstance(styles, list)
```

---

## 13. Phase 12: Documentation

### 13.1 Update Existing Documentation

**Files to update:**

1. `requirements.txt` - Add new dependencies
2. `.gitignore` - Add reports/, analysis/, .env
3. `PROJECT_DOCUMENTATION.md` - Add new sections

### 13.2 Create New Documentation

1. `ARCHITECTURE.md` - System architecture (created separately)
2. `IMPLEMENTATION.md` - This file

---

## 14. Appendix: Complete File Listing

### New Files to Create

```
video-amd-main/
├── services/
│   ├── __init__.py
│   ├── hf_client.py
│   ├── image_analyzer.py
│   ├── scene_aggregator.py
│   ├── prompt_loader.py
│   ├── report_generator.py
│   ├── report_cache.py
│   └── pdf_generator.py
├── prompts/
│   ├── formal.txt
│   ├── sarcastic.txt
│   ├── humorous_tech.txt
│   ├── humorous_non_tech.txt
│   └── jargon.txt
├── tests/
│   ├── __init__.py
│   ├── test_hf_client.py
│   ├── test_image_analyzer.py
│   ├── test_scene_aggregator.py
│   └── test_prompt_loader.py
├── .env.example
├── ARCHITECTURE.md
└── IMPLEMENTATION.md
```

### Existing Files to Modify

```
video-amd-main/
├── config.py          # Add HF config keys
├── requirements.txt   # Add new dependencies
├── .gitignore         # Add new directories
├── main.py            # Add report generation
└── app.py             # Add report UI
```

---

## Implementation Checklist

- [ ] Phase 1: Foundation Setup
  - [ ] Update config.py
  - [ ] Update requirements.txt
  - [ ] Create directory structure
  - [ ] Create .env.example

- [ ] Phase 2: HF Client Service
  - [ ] Create services/hf_client.py
  - [ ] Test basic connectivity

- [ ] Phase 3: Image Analysis Service
  - [ ] Create services/image_analyzer.py
  - [ ] Test JSON parsing

- [ ] Phase 4: Scene Aggregation
  - [ ] Create services/scene_aggregator.py
  - [ ] Test merging logic

- [ ] Phase 5: Prompt Templates
  - [ ] Create prompts/*.txt files
  - [ ] Create services/prompt_loader.py

- [ ] Phase 6: Report Generation
  - [ ] Create services/report_generator.py

- [ ] Phase 7: Report Caching
  - [ ] Create services/report_cache.py

- [ ] Phase 8: PDF Generation
  - [ ] Create services/pdf_generator.py

- [ ] Phase 9: CLI Integration
  - [ ] Update main.py

- [ ] Phase 10: Streamlit UI Integration
  - [ ] Update app.py

- [ ] Phase 11: Testing
  - [ ] Create test files
  - [ ] Run tests

- [ ] Phase 12: Documentation
  - [ ] Update PROJECT_DOCUMENTATION.md
  - [ ] Create ARCHITECTURE.md
  - [ ] Create IMPLEMENTATION.md

---

*Document Version: 1.0*
*Last Updated: 2026-07-08*
*Estimated Implementation Time: 5-6 hours*
