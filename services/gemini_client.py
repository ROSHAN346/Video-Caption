"""Gemini AI API Client (OpenAI-compatible endpoint)."""

import base64
import logging
from pathlib import Path
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"GeminiClient initialized: {base_url}")

    def analyze_image(self, image_path: str, prompt: str, model: str, max_tokens: int = 1024) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        suffix = path.suffix.lower()
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
        mime_type = mime_map.get(suffix, "jpeg")
        b64 = base64.b64encode(path.read_bytes()).decode()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{b64}"}}
        ]}]
        logger.info(f"Vision call: {path.name} -> {model}")
        r = self.client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content or ""

    def analyze_images_batch(self, image_paths: list[str], prompt: str, model: str, max_tokens: int = 2048) -> str:
        """One multimodal call with multiple images."""
        content: list = [{"type": "text", "text": prompt}]
        mime_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp"}
        for p in image_paths:
            path = Path(p)
            mime_type = mime_map.get(path.suffix.lower(), "jpeg")
            b64 = base64.b64encode(path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/{mime_type};base64,{b64}"}})
        messages = [{"role": "user", "content": content}]
        logger.info(f"Batch vision call: {len(image_paths)} images -> {model}")
        r = self.client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content or ""

    def generate_text(self, prompt: str, model: str, system_prompt: Optional[str] = None, max_tokens: int = 2048) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        logger.info(f"Text call: {model}")
        r = self.client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content or ""


_gemini_client = None


def get_gemini_client() -> GeminiClient:
    global _gemini_client
    if _gemini_client is None:
        from config import GEMINI_API_KEY, GEMINI_BASE_URL
        _gemini_client = GeminiClient(GEMINI_API_KEY, GEMINI_BASE_URL)
    return _gemini_client
