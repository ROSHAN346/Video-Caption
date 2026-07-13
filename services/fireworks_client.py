"""
Fireworks AI API Client Service

Provides a unified interface for communicating with Fireworks AI
using OpenAI-compatible API for both vision and text models.
"""

import base64
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class FireworksClient:
    """Wrapper around Fireworks AI OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str = "https://api.fireworks.ai/inference/v1"):
        """
        Initialize the Fireworks client.

        Args:
            api_key: Fireworks AI API key
            base_url: Fireworks API base URL
        """
        if not api_key:
            raise ValueError(
                "FIREWORKS_API_KEY is required. "
                "Get yours at: https://fireworks.ai/account/api-keys"
            )

        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"FireworksClient initialized with base_url: {base_url}")

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

        response = self.client.chat.completions.create(
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
        max_tokens: int = 1024
    ) -> str:
        """
        Generate text using a language model.

        Args:
            prompt: User prompt
            model: Model identifier
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text (may be empty; callers should treat "" as failure)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {}
        # gpt-oss / deepseek-style reasoning models can spend the whole
        # max_tokens budget on internal reasoning and return EMPTY content
        # unless the reasoning effort is capped.
        if "gpt-oss" in model or "qwen3" in model:
            kwargs["extra_body"] = {"reasoning_effort": "low"}

        logger.info(f"Generating text with model: {model}")
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content or ""


# Singleton instances: one per role so vision and text can use different keys.
_vision_client: Optional[FireworksClient] = None
_text_client: Optional[FireworksClient] = None


def get_fireworks_client(api_key: str = None, base_url: str = None) -> FireworksClient:
    """
    Get or create the Fireworks VISION client singleton (key #1).

    Args:
        api_key: API key (uses config FIREWORKS_API_KEY if None)
        base_url: API base URL (uses config default if None)

    Returns:
        FireworksClient instance
    """
    global _vision_client

    if _vision_client is None:
        from config import FIREWORKS_API_KEY, FIREWORKS_BASE_URL

        key = api_key or FIREWORKS_API_KEY
        url = base_url or FIREWORKS_BASE_URL

        _vision_client = FireworksClient(key, url)

    return _vision_client


def get_fireworks_text_client(api_key: str = None, base_url: str = None) -> FireworksClient:
    """
    Get or create the Fireworks TEXT client singleton (key #2).

    Uses FIREWORKS_TEXT_API_KEY, which falls back to FIREWORKS_API_KEY when
    a second key is not configured.

    Args:
        api_key: API key (uses config FIREWORKS_TEXT_API_KEY if None)
        base_url: API base URL (uses config default if None)

    Returns:
        FireworksClient instance
    """
    global _text_client

    if _text_client is None:
        from config import FIREWORKS_TEXT_API_KEY, FIREWORKS_TEXT_BASE_URL

        key = api_key or FIREWORKS_TEXT_API_KEY
        url = base_url or FIREWORKS_TEXT_BASE_URL

        _text_client = FireworksClient(key, url)

    return _text_client
