"""Gemini provider. Everything that knows about the google-genai SDK lives in this package."""

from app.llm.providers.gemini.adapter import GeminiProvider

__all__ = ["GeminiProvider"]
