"""
Fireworks AI API Client Service

Provides a unified interface for communicating with Fireworks AI
using OpenAI-compatible API for both vision and text models.
"""

import base64
import logging
from pathlib import Path
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
            model: Model identifier (e.g., accounts/fireworks/models/llama-v3p2-11b-vision-instruct)
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
        response = self.client.chat.completions.create(
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
        response = self.client.chat.completions.create(
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
            response = self.client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


# Singleton instance for convenience
_client_instance: Optional[FireworksClient] = None


def get_fireworks_client(api_key: str = None, base_url: str = None) -> FireworksClient:
    """
    Get or create Fireworks client singleton.

    Args:
        api_key: API key (uses config default if None)
        base_url: API base URL (uses config default if None)

    Returns:
        FireworksClient instance
    """
    global _client_instance

    if _client_instance is None:
        from config import FIREWORKS_API_KEY, FIREWORKS_BASE_URL

        key = api_key or FIREWORKS_API_KEY
        url = base_url or FIREWORKS_BASE_URL

        _client_instance = FireworksClient(key, url)

    return _client_instance
