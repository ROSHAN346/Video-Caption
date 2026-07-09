"""Groq AI API Client (text only, OpenAI-compatible)."""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class GroqClient:
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"GroqClient initialized: {base_url}")

    def generate_text(self, prompt: str, model: str, system_prompt: str = None, max_tokens: int = 1024) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        r = self.client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        return r.choices[0].message.content or ""


_groq_client = None


def get_groq_client() -> GroqClient:
    global _groq_client
    if _groq_client is None:
        from config import GROQ_API_KEY, GROQ_BASE_URL
        _groq_client = GroqClient(GROQ_API_KEY, GROQ_BASE_URL)
    return _groq_client
